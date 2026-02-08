---
title: "Gemini Diffusion"
description: "Why diffusion-style generation could reshape how agents think."
pubDate: 2025-05-30
tags: ["research", "models", "agents"]
icon: "✨"
---

Diffusion is usually associated with images, but the idea of iterative refinement is powerful for agents. Instead of jumping to a final answer, a diffusion-style process can explore a space of options and then converge.

For agent workflows, this suggests two opportunities:

- Generate multiple candidate plans, then compress to the best one.
- Make intermediate states inspectable so humans can steer.

I care less about model branding and more about behavior. If the agent can show me “here are the three directions I explored,” I get a safer and more collaborative experience.

The product question becomes: how do we surface this without overwhelming the user? My bias is to show the reasoning only when asked, but keep it available for audit.
