# Codex Skill: Frontend (Astro)

Goal: keep the site minimal, fast, and content-first.

Rules:
- Prefer Astro pages + content collections.
- Avoid heavy client JS; default to server/static rendering.
- Keep CSS simple, typography-first, minimal components.
- Current design direction: **centered mono layout** inspired by niels.degran.de.
  - Use IBM Plex Mono.
  - Single accent hue (`--accent-color`) applied consistently to:
    - nav links
    - all page titles (Home/About/Blog)
    - blog post titles (in list + inside post pages)
    - headings (h1/h2/h3)
  - Links should feel designed (hover treatment), but **no underlines**.
  - Blog list items should include a small leading **icon** before the post title.
  - Tagline under name: "Senior AI Engineer · Paris".
  - Validate by visually checking all pages before sending screenshots.

Commands:
- cd site
- npm install
- npm run dev
- npm run build

Definition of done:
- `npm run build` passes.
- Blog list + post pages render from markdown.
- RSS works.
