---
title: "Agent builders"
description: "The real product is not the model, it is the builder experience."
pubDate: 2025-07-22
tags: ["product", "agents", "platform"]
icon: "🧱"
---

I keep seeing platforms chase model quality while neglecting the builder loop. Builders want a tight feedback cycle: change prompt, run test, see diff.

A good agent builder has:

- A schema for tools and data that is easy to reason about.
- First-class evaluation with golden tasks.
- A clear path from prototype to production (auth, rate limits, logs).

The best builders also encode opinion. They nudge you into safe defaults and keep the edge cases visible. That saves months of “why is this brittle” debugging.

If I were designing this, the north star would be time-to-first-correct-run. Everything that helps that metric is worth doing.
