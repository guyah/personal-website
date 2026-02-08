---
title: "Rehosting"
description: "Why moving an agent stack is a product decision, not just infra work."
pubDate: 2025-09-12
tags: ["infrastructure", "product", "systems"]
icon: "🏗️"
---

Rehosting looks simple on a diagram: lift, shift, done. In practice, it is a re-negotiation of latency, reliability, and cost, and those choices leak into user experience.

When I evaluate a rehost, I map it to three questions:

- What breaks if the agent responds 300 ms slower?
- Which failures are user-visible versus recoverable?
- What does observability look like on day one?

A clean rehost has a thin shim that preserves the contract. Anything that changes the contract should be treated like a feature release, with beta users and a rollback path.

The smallest win is lower cloud bills. The real win is owning the control plane so I can build better routing, smarter retries, and more deterministic behavior over time.
