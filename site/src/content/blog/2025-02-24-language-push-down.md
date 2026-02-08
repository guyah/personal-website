---
title: "Language push down"
description: "Push language into the system so the agent can reason higher up."
pubDate: 2025-02-24
tags: ["systems", "agents", "architecture"]
icon: "⬇️"
---

I like the idea of pushing language “down” the stack. Instead of forcing the model to parse messy tool outputs, make the tools speak in structured, readable terms.

This has a few practical outcomes:

- Prompts get shorter and clearer.
- The agent makes fewer brittle assumptions.
- Observability improves because the tool output is human-legible.

In other words, the system does more of the hard work, so the model can focus on the decision layer.

The rule I use: if a human would struggle to read a tool response, the model will struggle too.
