---
title: "Agent management"
description: "Managing agents is closer to SRE than to chatbot ops."
pubDate: 2025-05-05
tags: ["operations", "agents", "systems"]
icon: "📈"
---

Once an agent is in production, the real work begins: uptime, drift, cost, and safety. I treat agent management like reliability engineering.

What matters most:

- Versioned prompts and tools.
- Clear rollout and rollback paths.
- Alerting on behavior, not just latency.

I also keep a weekly audit of “weird outputs.” Those are early signals of drift or data issues.

A managed agent should be boring. If it surprises you often, it is not ready for critical tasks.
