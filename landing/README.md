# DistillForge Cloud — Landing page

Single-file static landing page for the **DistillForge Cloud** offer
(`https://<task>.<tenant>.df.arcamens.ai`). No build step, no dependencies.

## Files
- `index.html` — the whole page (HTML + inline CSS + inline SVG logo).
- `netlify.toml` — Netlify config (publish dir, security headers, cache).

## Local preview
Open `index.html` in a browser, or:
```sh
cd landing && python3 -m http.server 8000   # then visit http://localhost:8000
```

## Deploy on Netlify (from GitHub)
1. Push this repo to GitHub.
2. In Netlify: **Add new site → Import from GitHub**, pick the repo.
3. Set **Base directory** = `landing`, **Publish directory** = `landing`
   (leave Build command empty).
4. Deploy. Point a domain such as `cloud.arcamens.ai` at the site.

## Notes
- Copy is English (matches the brandbook + default README). A French variant
  can live at `index.fr.html` if needed.
- The logo is the brandbook logomark inlined as SVG (no heavy PNG over the wire).
- GitHub links point to `https://github.com/ArcamensAI/distillforge`.
- Update the contact address (`hello@arcamens.ai`) before going live.
