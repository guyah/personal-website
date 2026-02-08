# Codex Skill: Frontend (Astro)

Goal: keep the site minimal, fast, and content-first.

## Design direction (CURRENT)
Match the look & feel of **davekiss.com** (warm paper light theme + espresso dark theme).

Source of truth:
- `.codex/skills/style-davekiss.md` (tokens + rules)

## Where styles live
- Global CSS is defined in: `site/src/layouts/Layout.astro` using `<style is:global>`.
- Prefer implementing all tokens via CSS variables at `:root` and only lightly styling pages.

## Rules
- Prefer Astro pages + content collections.
- Avoid heavy client JS; default to server/static rendering.
- Typography-first; minimal components.
- Respect `prefers-color-scheme` (light/dark).
- Use **serif headings** + **sans body**; generous line-height.
- Nav: centered, uppercase links with letter-spacing.
- Lists (blog index): subtle borders + warm surface; serif titles.

## Conventions
- If you add list styles, reuse shared classes (e.g. `.list`, `.list-item`) defined globally.
- Keep pages (`index.astro`, `about.astro`, `pages/blog/index.astro`) mostly semantic HTML; avoid per-page bespoke CSS unless necessary.

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
