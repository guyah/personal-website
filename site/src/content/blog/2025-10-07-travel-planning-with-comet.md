---
title: "Travel planning with Comet"
description: "Using a voice-first agent to turn messy trip inputs into a tight itinerary."
pubDate: 2025-10-07
tags: ["agents", "product", "workflows"]
icon: "🧭"
---

I keep a mental model for travel planning that looks like a backlog: inputs are chaotic, constraints are hidden, and the output needs to feel simple. A good agent does not just search. It reconciles tradeoffs and keeps the plan editable.

My preferred flow is a two-pass agent:

- Pass 1: extract constraints and preferences (time windows, budget bands, walking tolerance, "must-see" vs "nice-to-have").
- Pass 2: assemble options, then compress to a narrative itinerary with explicit assumptions.

The voice layer matters more than I expected. Spoken prompts turn indecision into structured input. A small “say it out loud” mode collects details faster than a form and gives the agent the right priors.

What makes Comet-style planning work is explicitness:

- Every decision is annotated with why it was chosen.
- Gaps are surfaced as questions, not hidden.
- The plan is editable without redoing the whole search.

If I were shipping this today, I would add a lightweight “change one thing” command. That is the real test of whether the plan is an object the user owns, not a blob the agent produced.
