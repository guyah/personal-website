---
title: "A(gent)ware"
description: "Agentware is the operating layer between models and real work."
pubDate: 2025-05-13
tags: ["platform", "agents", "infrastructure"]
icon: "🧠"
---

I use “agentware” to describe the stack around the model: memory, tools, routing, logging, and policy. It is the difference between a demo and a product.

The minimal agentware stack includes:

- Tooling with typed inputs and predictable errors.
- Memory with clear retention policies.
- Observability across prompts, tools, and outputs.

This layer is where most engineering time should go. Models improve quickly, but product reliability comes from the surrounding system.

If you build agentware well, you can swap models without rewriting everything. That is leverage.
