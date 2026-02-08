---
title: "Smart-ECG-diagnosis: a field-first ECG diagnosis build"
description: "A concise build story for an ECG diagnosis system with real-time intent and a working demo."
pubDate: 2026-02-08
tags: ["ml", "health", "signals", "python", "research"]
icon: "🫀"
---

## Hook

I built **Smart-ECG-diagnosis** to move ECG diagnosis from lab curiosity to field-ready flow. The work centers on fast, reliable automated ECG with a demo you can actually run.

## Context

Check My Beat is a collaboration with the Lebanese Red Cross to help medical operators make faster real-time decisions for critical heart conditions and defibrillation. The goal is simple: turn ECG signals into dependable, automated diagnosis without friction.

## Methodology

Start from the operational need, then back into the system design. I treated this as a product problem first and a model problem second:

- keep the signal path short and predictable
- automate the diagnostic step instead of adding more manual review
- make the flow real-time by design, not by afterthought

## Implementation notes

The project is split by responsibility so it can be deployed and tested in parts:

- **Unity front end plus microservices** for the operator interface and system coordination.
- **Python backend services** for MongoDB persistence and a k-means clustering pipeline.
- **Clear module boundaries** so the front end and backend can evolve without blocking each other.

## Findings

The work produced a working demo and a structured codebase that maps to the real workflow. That structure made it clear which parts belong to the UI flow and which belong to data handling and clustering.

## Impact

This project is built to help medical operators deliver fast, reliable ECG diagnosis with automated support for critical decisions and defibrillation contexts. It is focused on speed, repeatability, and clearer handoff from signal to decision.

## What I would do next

- validate the end to end path with a broader set of ECG conditions
- harden deployment for field use with offline and failover behavior
- add operator feedback loops to improve diagnostic routing over time

## Links

- GitHub: https://github.com/guyah/Smart-ECG-diagnosis
- Demo video: https://www.youtube.com/watch?v=r3BZmdGSI4o
