use chrono::Utc;
use serde_json::json;
use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

#[derive(Debug, Clone)]
struct WorkerConfig {
    listen_addr: String,
    model_id: String,
    response_text: String,
    confidence: f64,
}

fn main() -> std::io::Result<()> {
    env_logger::init();

    let config = WorkerConfig {
        listen_addr: env::var("DISTILLFORGE_STUDENT_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:9100".to_string()),
        model_id: env::var("DISTILLFORGE_STUDENT_MODEL")
            .unwrap_or_else(|_| "local_student".to_string()),
        response_text: env::var("DISTILLFORGE_STUDENT_RESPONSE")
            .unwrap_or_else(|_| "student ok".to_string()),
        confidence: env::var("DISTILLFORGE_STUDENT_CONFIDENCE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.99),
    };

    let listener = TcpListener::bind(&config.listen_addr)?;
    log::info!(
        "DistillForge student worker listening on {}",
        config.listen_addr
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_connection(stream, &config) {
                    log::warn!("student worker connection failed: {err}");
                }
            }
            Err(err) => log::warn!("student worker accept failed: {err}"),
        }
    }

    Ok(())
}

fn handle_connection(mut stream: TcpStream, config: &WorkerConfig) -> std::io::Result<()> {
    let started_at = Instant::now();
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;

    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return write_response(&mut stream, 400, "text/plain", b"bad request\n");
    }

    let method = parts[0];
    let path = parts[1];
    let content_length = read_headers(&mut reader)?;
    if content_length > 0 {
        let mut body = vec![0; content_length];
        reader.read_exact(&mut body)?;
    }

    match (method, path) {
        ("GET", "/health") => write_response(&mut stream, 200, "text/plain", b"ok\n"),
        ("POST", "/infer") => {
            let body = infer_body(config, started_at.elapsed().as_millis());
            write_json(&mut stream, &body)
        }
        ("POST", "/v1/chat/completions") => {
            let body = chat_completion_body(config);
            write_json(&mut stream, &body)
        }
        ("POST", "/v1/completions") => {
            let body = completion_body(config);
            write_json(&mut stream, &body)
        }
        _ => write_response(&mut stream, 404, "text/plain", b"not found\n"),
    }
}

fn read_headers(reader: &mut BufReader<TcpStream>) -> std::io::Result<usize> {
    let mut content_length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = value.trim().parse().unwrap_or(0);
        } else if let Some(value) = line.strip_prefix("content-length:") {
            content_length = value.trim().parse().unwrap_or(0);
        }
    }
    Ok(content_length)
}

fn infer_body(config: &WorkerConfig, latency_ms: u128) -> serde_json::Value {
    json!({
        "model_id": config.model_id,
        "output": config.response_text,
        "confidence": config.confidence,
        "latency_ms": latency_ms
    })
}

fn chat_completion_body(config: &WorkerConfig) -> serde_json::Value {
    json!({
        "id": format!("chatcmpl_student_{}", Utc::now().timestamp_millis()),
        "object": "chat.completion",
        "model": config.model_id,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": config.response_text
                },
                "finish_reason": "stop"
            }
        ]
    })
}

fn completion_body(config: &WorkerConfig) -> serde_json::Value {
    json!({
        "id": format!("cmpl_student_{}", Utc::now().timestamp_millis()),
        "object": "text_completion",
        "model": config.model_id,
        "choices": [
            {
                "index": 0,
                "text": config.response_text,
                "finish_reason": "stop"
            }
        ]
    })
}

fn write_json(stream: &mut TcpStream, body: &serde_json::Value) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(body).expect("student response should serialize");
    write_response(stream, 200, "application/json", &bytes)
}

fn write_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> WorkerConfig {
        WorkerConfig {
            listen_addr: "127.0.0.1:0".to_string(),
            model_id: "local_student".to_string(),
            response_text: "student ok".to_string(),
            confidence: 0.99,
        }
    }

    #[test]
    fn builds_infer_response() {
        let body = infer_body(&test_config(), 12);

        assert_eq!(body["model_id"], "local_student");
        assert_eq!(body["output"], "student ok");
        assert_eq!(body["confidence"], 0.99);
        assert_eq!(body["latency_ms"], 12);
    }

    #[test]
    fn builds_chat_completion_response() {
        let body = chat_completion_body(&test_config());

        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "local_student");
        assert_eq!(body["choices"][0]["message"]["content"], "student ok");
    }
}
