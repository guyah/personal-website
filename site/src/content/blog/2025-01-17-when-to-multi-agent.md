---
title: "When to multi-agent"
description: "A quick decision framework for splitting work across agents."
pubDate: 2025-01-17
tags: ["agents", "architecture", "workflow"]
icon: "🤝"
---

I reach for a multi-agent setup only when a single agent starts to feel overloaded. There is a coordination tax that you should only pay when it buys clarity.

A simple heuristic:

- Parallelizable tasks with minimal shared state? Use multiple agents.
- Heavy shared context or strict ordering? Keep it single.

I also consider failure isolation. If a task has a high error cost, splitting it into specialists can make review easier.

Multi-agent is not a flex. It is a tool. Use it when it reduces risk or time-to-clarity.
