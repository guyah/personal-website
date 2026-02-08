---
title: "Reviewing generated code"
description: "A pragmatic review checklist for agent-written diffs."
pubDate: 2025-06-26
tags: ["engineering", "devtools", "agents"]
icon: "🔍"
---

I review agent-generated code as if it came from a new teammate: kind, skeptical, and focused on risks. The code might compile, but will it behave?

My quick checklist:

- Do the tests actually cover the new behavior?
- Are errors handled or just logged?
- Is the change reversible if something breaks?

I also scan for “quiet complexity” like new dependencies, hidden timeouts, or broad data access. These are the places where agents tend to be too bold.

The best improvement I have made is to require the agent to describe the diff in plain language before I review it. If it cannot explain it, I should not merge it.
