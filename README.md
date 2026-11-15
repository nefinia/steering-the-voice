# Steering the voice

Chat training gives language models one shared assistant voice. In [Epistemic Bridges](https://github.com/nefinia/epistemic-bridges), people and models linked the same pairs of simple ideas, and most of what made chat answers look narrow against the human answers was that voice. Asking a model to answer as an ordinary person drawn at random widened its ideas on every model.

This repository asks whether the same can be done inside the model, with no prompt: find the direction of the assistant voice in the model's activations and steer it, then measure the answers against the map that real people drew.

Built for the Lexsi Labs AI Safety and Alignment Hackathon, November 2026, with [SafeTune](https://github.com/Lexsi-Labs/SafeTune).

## What it does

On Qwen2.5-7B-Instruct, the model with the clearest change from base to chat in Bridges, two steering vectors are taken with contrastive activation addition (Panickssery et al. 2023, as implemented in SafeTune), at the middle layers (8 to 19 of 28):

- **life:** the Bridges prompt with an ordinary life as the system message, minus the same prompt with none, at the last prompt token. It holds who is answering, before a word is written.
- **voice:** the model's own answers given as an ordinary life, minus its own answers with no persona, averaged over the answer. It holds how the answer is written.

Each vector is added at four strengths (0.25, 0.5, 1 and 2) while the model answers the six core pairs of Bridges, in both directions, ten times each, with no system message. An unsteered run is the reference.

## How it is measured

With the measures of Epistemic Bridges, against the answers of the players (which stay private and are not in this repository):

- **recall:** how much of the human map of ideas the answers reach;
- **precision:** how many of the answers land on it;
- **voice:** how much the answers repeat their sentence patterns.

The question is whether steering widens the ideas the way the ordinary-life prompt does, and at what strength the answers start to break.

## In this repo

- `steer_modal.py`: extracts the vectors and generates the steered answers on Modal (one L40S GPU).
- `data/`: the pairs and instructions of the game, the ordinary lives, and the model's own answers from Bridges used to build the voice vector.
- `results/`: the steered answers, written by the workflow.
- `.github/workflows/run.yml`: runs it from the Actions tab. It needs two repository secrets, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

```
pip install modal
modal run steer_modal.py
```

## Status

First run in progress.

Sofia Gallego · CosmoVentures · Paris
