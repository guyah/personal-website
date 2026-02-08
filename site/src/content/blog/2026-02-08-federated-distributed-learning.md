---
title: "Federated and Distributed Learning from scratch"
description: "A PyTorch project where identical workers train on local batches, send gradients to a server, and learn from an averaged update."
pubDate: 2026-02-08
tags: ["federated-learning", "pytorch", "distributed", "ml", "systems"]
icon: "🧠"
---

## TL;DR

I built a small federated learning system in PyTorch. Multiple workers train on their own batches, send gradients to a server, get back an averaged gradient, and update in sync. The point was to learn the full loop and show how this scales to huge datasets with limited nodes.

## Context

I wanted a clean, modular setup to explore federated learning and distributed computation without hiding behind a big framework. The core idea is simple: keep data local, keep models identical, and learn from an aggregate signal.

## What I built

A basic federated training pipeline with a server and multiple workers:

- each worker holds a local batch of samples
- each worker runs a forward pass, then computes gradients
- the server aggregates gradients by averaging across workers
- workers update their parameters using the averaged gradient

All models are identical but can start from different initializations. The result is training that optimizes performance across multiple data instances, not just one local shard.

## Key decisions

- **Identical model architecture on every worker.** This keeps aggregation straightforward and makes the averaged gradient meaningful.
- **Server-side gradient averaging.** It mirrors the classic federated learning flow and keeps workers lightweight.
- **Modular layout.** The goal was to make the system easy to extend, not just get a single run working.
- **PyTorch implementation.** It is the most direct way to express the compute graph and gradient flow.

## Lessons

- Aggregation is simple to describe but easy to get wrong in code. Getting the order of operations right matters.
- Averaging gradients is a clean baseline that already shows how multiple data holders can collaborate.
- Federated setups are a good fit for huge datasets, especially when each node has limited compute.

## Links

- Repo: https://github.com/guyah/Federated-and-Distributed-Learning
