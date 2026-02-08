---
title: "Misbehaving LLMs"
description: "Failure modes I plan for before an agent touches production."
pubDate: 2025-06-27
tags: ["reliability", "agents", "safety"]
icon: "⚠️"
---

Most agent incidents are not spectacular. They are subtle: a silent assumption, a stale tool response, a missing edge case. That is why I focus on small, repeatable failure modes.

My short list:

- Overconfident actions without sufficient evidence.
- Tool timeouts that result in partial outputs.
- Context drift across long sessions.

The fix is a mix of guardrails and design:

- Require citations for any action that changes state.
- Build a “no-result” response path that is acceptable to users.
- Reset or summarize context on purpose, not by accident.

I do not expect perfect behavior. I expect predictable behavior. The difference is enormous.
