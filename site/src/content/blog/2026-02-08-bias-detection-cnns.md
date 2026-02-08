---
title: "Bias-Detection-in-CNNs: building a small lab for model bias checks"
description: "Why I built a focused repo to probe bias in CNNs, what I chose to keep simple, and what I learned from the first runs."
pubDate: 2026-02-08
tags: ["ml", "vision", "fairness", "research", "cnn"]
icon: "🧠"
---

## TL;DR

I built **Bias-Detection-in-CNNs** as a compact, repeatable lab for checking bias signals in convolutional models. It is a place to run targeted probes, compare behavior shifts, and keep notes on what actually changes performance.

## Context

CNNs can look great on benchmark accuracy and still lean on shortcuts. I wanted a repo that makes it easy to ask one question at a time: **what bias signal is the model using, and how stable is it when I perturb the input**.

## What I built

- a focused codebase for bias probes in CNNs
- small experiments that isolate one bias signal per run
- a thin evaluation layer so results are comparable across runs
- lightweight notes so the why is not lost later

## Key decisions

- **Keep runs small.** Fast experiments beat perfect experiments when you are still mapping the space.
- **Prefer probes over theory.** I care about what breaks first, not what should break.
- **Structure for repeatability.** If I cannot rerun it in an hour, it does not belong.

## Lessons

- Bias is rarely a single knob. You see it in interactions, not in isolation.
- A tiny, well scoped repo makes iteration faster than a big kitchen sink project.
- Most insights came from failures, not the clean runs.

## Links

- GitHub: https://github.com/guyah/Bias-Detection-in-CNNs
