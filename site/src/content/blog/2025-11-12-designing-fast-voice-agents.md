---
title: Designing fast voice agents
description: What matters most when latency, turn-taking, and reliability meet real users.
pubDate: 2025-11-12
tags: [voice, systems]
---

Fast voice agents live or die by the gaps between words. When the pauses feel human, users relax and keep talking. When they feel brittle, every interruption becomes a drop.

A few principles guide my approach:

1. **Cut the long tail.** Users tolerate an average latency if the worst-case never surprises them. Measure p95 and p99, then design around them.
2. **Stream everything.** Partial transcription and partial synthesis allow the system to stay present.
3. **Own the orchestration.** The glue between the model, the audio stack, and the product UX is where the experience is won.

I treat latency like a product feature: visible, monitored, and constantly tuned.
