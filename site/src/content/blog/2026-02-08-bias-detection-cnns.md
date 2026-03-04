---
title: "Bias-Detection-in-CNNs: building a bias lab with Colored MNIST and CelebA"
description: "A concise bias-check lab for CNNs using Colored MNIST and CelebA, with focused probes, repeatable runs, and clear takeaways."
pubDate: 2026-02-08
tags: ["ml", "vision", "fairness", "research", "cnn"]
icon: "🧠"
---

## Hook

I wanted a tiny lab where bias signals in CNNs are the main character. No sprawling framework, just a repeatable way to ask one question at a time and see what the model actually uses.

## Context

Benchmarks hide shortcuts. A model can score well while leaning on unintended cues. This repo is my controlled environment for probing those cues in vision models using Colored MNIST and CelebA.

## Methodology

- Start with datasets where spurious signals are easy to surface.
- Isolate one bias signal per run so behavior changes are attributable.
- Compare runs using the same evaluation flow to keep the delta honest.

## Implementation notes

The repo focuses on a small set of concrete implementations:

- CelebA color splitter to isolate RGB channels in the dataset.
- MNIST bias detection workflows using colored variants.
- CelebA training on full RGB.
- CelebA training on individual channels.

Each piece is deliberately narrow. The point is to test bias assumptions without extra moving parts.

## Findings

I did not chase benchmark numbers here. The validation was qualitative and behavioral:

- When color channels are separated, model behavior shifts in ways that expose reliance on color cues.
- Training on full RGB versus isolated channels changes what the model appears to attend to.
- Small, controlled probes surface bias signals faster than large end to end pipelines.

## Impact

This lab gives me a repeatable way to stress a CNN and see which signals it leans on. It also keeps the experiments small enough to iterate quickly and write down what I learn while the context is fresh.

## What I would do next

- Add a minimal quantitative layer for comparing runs without turning it into a benchmark race.
- Expand the probe set beyond color to other controlled spurious cues.
- Package the probes into a simple CLI so I can run a new bias check in minutes.

## Links

- GitHub: https://github.com/guyah/Bias-Detection-in-CNNs

## References

- CelebA paper: "Deep Learning Face Attributes in the Wild" (Liu et al.) https://arxiv.org/abs/1411.7766
- Colored MNIST as a bias testbed shows up in multiple shortcut learning papers, for example: "Invariant Risk Minimization" (Arjovsky et al.) https://arxiv.org/abs/1907.02893
