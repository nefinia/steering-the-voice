---
license: cc-by-nc-4.0
language: [en]
tags: [steering, epistemic-diversity, tiny-aya, safetune, auditkit]
---
# Steering the voice: Tiny Aya answers and measures

AI answers from the Lexsi AI Safety and Alignment Hackathon project
[steering-the-voice](https://github.com/nefinia/steering-the-voice): Tiny Aya Base, Tiny Aya Global plain,
Tiny Aya Global answering as an ordinary life, and Tiny Aya Global steered with SafeTune toward that
perspective, on the six core pairs of [Epistemic Bridges](https://github.com/nefinia/epistemic-bridges).

- `tiny-aya_steer.jsonl`: every answer, with its condition, steering vector and strength.
- `measures.csv`: recall and precision on the human map of ideas, and how much the voice repeats its sentence patterns.
- `auditkit.json`: AuditKIT checks on the plain and steered model (general knowledge, refusal of harmful requests, over-refusal).
- `steering_vectors.json`: layers and norms of the vectors.

The answers of the human players are private and are not included. The model outputs follow the
CC-BY-NC licence of Tiny Aya.
