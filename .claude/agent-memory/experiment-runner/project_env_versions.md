---
name: Experiment environment baseline versions
description: Python, SDK, and dependency versions observed during experiments -- reference for reproducibility
type: project
---

As of 2026-03-16:
- Python 3.12.9 (pyenv virtualenv `cognition`)
- openai SDK 2.26.0
- pydantic 2.12.5
- OS: Fedora 42, Linux 6.19.7-100.fc42.x86_64

**Why:** Version drift can cause different experiment outcomes. Recording baselines helps diagnose future discrepancies.

**How to apply:** When an experiment fails unexpectedly, compare current versions against these baselines to check for regressions.
