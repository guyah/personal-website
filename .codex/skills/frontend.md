# Codex Skill: Frontend (Astro)

Goal: keep the site minimal, fast, and content-first.

Rules:
- Prefer Astro pages + content collections.
- Avoid heavy client JS; default to server/static rendering.
- Keep CSS simple, typography-first, minimal components.
- Current design direction: **centered mono layout** inspired by niels.degran.de.
  - Use IBM Plex Mono.
  - Single accent hue (`--accent-color`) applied to **links + headings**.
  - Body text stays near-black for readability; minimal borders/decoration.

Commands:
- cd site
- npm install
- npm run dev
- npm run build

Definition of done:
- `npm run build` passes.
- Blog list + post pages render from markdown.
- RSS works.
