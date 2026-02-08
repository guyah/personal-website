---
title: "Incident agent"
description: "Designing a responder that helps humans, not a bot that adds noise."
pubDate: 2025-08-04
tags: ["reliability", "agents", "operations"]
icon: "🚑"
---

An incident agent is only useful if it reduces paging fatigue. That means it has to be a calm teammate that summarizes, not a frantic narrator.

My baseline requirements:

- It ingests logs, traces, and alerts into one timeline.
- It provides a single hypothesis with evidence, not a list of guesses.
- It calls out missing data explicitly.

The best pattern I have seen is “triage first, automate later.” Let the agent create a shared incident brief, then suggest next steps. Once those steps are predictable, automate the mechanical ones.

I also want human handoff to be trivial. One command should export a concise incident summary that can be pasted into a postmortem doc.

If we do this well, we are not replacing on-call judgment. We are compressing the time to shared context so the team can focus on the fix.
