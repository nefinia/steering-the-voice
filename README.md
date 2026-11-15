# Steering the voice

**Steering Tiny Aya's assistant voice toward the range of ideas real people reach, with no prompt and no retraining.**

## Tracks and problem statements

- Track 2b, AI Safety and Correction
  - Other: inference-time steering for epistemic diversity (next to problem statement "Inference-time safety steering", with the same method applied to a different behaviour)

## Abstract

Chat training gives language models one shared assistant voice. In Epistemic Bridges, people and 15 open models linked the same pairs of words with no evident connection, and the players' answers drew a map of where people go. Chat models reached less of that map than people, even with the voice removed from every answer, and the one prompt that widened every model's ideas was answering as an ordinary person drawn at random.

This project asks whether the same can be done inside the model, with no prompt and no retraining, on Tiny Aya Global (3.35B). With SafeTune, two contrastive activation addition vectors are taken at the middle layers: a "life" vector from the prompt with and without an ordinary life as system message, and a "voice" vector from the model's own answers given as a life against its own plain answers. Each is added at three strengths while the model answers the six core Bridges pairs in both directions, ten times each, with no system message. Tiny Aya Base gives the reference from before chat training.

The answers are measured against the human map with the Bridges measures, which use no model as judge: recall (how much of the human map the answers reach), precision (how many answers land on it), and how much the voice repeats its sentence patterns. AuditKIT checks what should not get worse on the plain and steered model: general knowledge (Global-MMLU-Lite), refusal of harmful requests (MultiJail) and over-refusal of safe requests (XSTest).

Chat training narrows Tiny Aya as it does other models: recall falls from 0.76 for the base model to 0.64 for the chat model, and the voice becomes more repetitive. The voice vector moves the chat model back. At strength 0.25, recall rises to 0.72 with precision unchanged (0.73) and a more varied voice, while general knowledge (0.45 against 0.46) and refusal of harmful requests (0.70 against 0.65) hold. At 0.5, recall reaches 0.81, the level of people, with precision 0.67, slightly above what the ordinary-life prompt gives (recall 0.77, precision 0.61); general knowledge is unchanged (0.46), refusal of harmful requests rises to 0.80, and over-refusal rises a little (0.05 to 0.10). At 1, the answers drift and both measures fall.

The life vector does not work this way. Even at the lowest strength it pushes the model into a caricature of "ordinary" speech, a thick invented dialect, and at higher strengths into repeated syllables, with recall near zero and every check failing. A direction built from who is speaking carries the model's stereotypes of those people; a direction built from how the model's own answers change carries the change in ideas. Steering for diversity has to be read against real people, because the model's idea of an ordinary person is itself narrow.

## Headline

**Recall on the human map of ideas, Tiny Aya Global: 0.64 → 0.72 with precision held (0.72 → 0.73) at voice strength 0.25, and 0.64 → 0.81 at 0.5, the level of people (0.81).** General knowledge and refusal of harmful requests hold at 0.25.

**Base model and method:** `CohereLabs/tiny-aya-global` (3.35B), steered at inference with contrastive activation addition from SafeTune, checked with AuditKIT.

## For judges: one minute

- Chat models answer in one shared assistant voice. In [Epistemic Bridges](https://github.com/nefinia/epistemic-bridges), people and 15 open models linked the same pairs of words, and chat models reached less of the space of ideas people reached, even with that voice removed.
- Asking a model to answer as an ordinary person drawn at random was the one prompt that widened its ideas on every model.
- Here that perspective is turned into a direction in Tiny Aya's activations with SafeTune, and added at inference with no system prompt.
- The answers are read against the map of ideas drawn by real players (recall and precision), and AuditKIT checks that general knowledge and safety hold.

## Problem statement and approach

### Why this problem

As more people think with AI, the ideas they meet are more and more the ones models give. Chat training makes models reliable, and also makes them sound alike: their answers to open questions share one voice and cluster around the same few ideas, across models and across people who use them. A model can be safe and still narrow how people think. Most safety tooling treats the assistant persona as the goal; this project treats it as a behaviour that can be measured and steered, like refusal.

### Approach

1. The six core pairs of Bridges (concrete to abstract), in both directions, sampled 10 times each from Tiny Aya Base (before chat training), Tiny Aya Global with no persona, and Tiny Aya Global answering as an ordinary life drawn from 30 short sketches.
2. SafeTune extracts two contrastive activation addition (CAA) vectors at the middle layers:
   - **life:** the prompt with an ordinary life as system message, minus the same prompt with none, at the last prompt token;
   - **voice:** the model's own answers as a life, minus its own plain answers, averaged over the answer.
3. SafeTune adds each vector at strengths 0.25, 0.5 and 1 while Tiny Aya Global answers with no system message.
4. Every answer is measured against the human map of ideas; AuditKIT runs the same checks on the plain and steered model.

### Research questions

- Does Tiny Aya's chat training narrow the ideas it reaches, as it does for other chat models?
- Can a steering vector move the chat model back toward the range of people, without a prompt?
- At what strength does steering start to break the answers, or to cost knowledge or safety?

### Related work

Contrastive activation addition (Panickssery et al. 2023, Steering Llama 2 via Contrastive Activation Addition); chat training reduces output diversity (Kirk et al. 2024, ICLR); models are homogeneously creative (Wenger and Kenett 2026, PNAS Nexus); open-ended homogeneity across models (Jiang et al. 2025, NeurIPS); precision and recall for generative models (Kynkäänniemi et al. 2019, NeurIPS); syntactic templates in generated text (Shaib et al. 2024, EMNLP).

## Hackathon repos used

- **SafeTune** (`safetune==0.1.21`): `extract_caa_vectors` with `CAAConfig(target_layers=..., pool_method="last_token" | "mean")` for the two vectors, and `CAAModel(model, vectors, strength=s)` to steer generation.
- **AuditKIT** (`auditkit[interop]`): `load_global_mmlu` (Global-MMLU-Lite, English), `load_multijail` (English) and `load_xstest` (safe prompts); `ak.evaluate` with the steered model passed as a callable, the `mcq` adapter, and scorers built on AuditKIT's multilingual refusal detector (`find_refusal`).

### How the repos connect

```
Bridges pairs ─► Tiny Aya Base / Global (plain, as a life) ─► SafeTune: extract life and voice vectors
                                                              │
                                                              ▼
                                   SafeTune CAAModel: steered Tiny Aya Global (0.25, 0.5, 1)
                                        │                                   │
                                        ▼                                   ▼
                     Bridges measures vs the human map         AuditKIT: knowledge, refusal, over-refusal
```

The same steered model object from SafeTune is what AuditKIT evaluates, so the safety and knowledge checks apply to exactly the model whose ideas are measured.

## Models and licenses

- **Base Aya model:** `CohereLabs/tiny-aya-global`, bfloat16, Hugging Face transformers, CC-BY-NC-4.0. Steered model; no weights changed.
- **Reference before chat training:** `CohereLabs/tiny-aya-base`, CC-BY-NC-4.0.
- Earlier pilot (before the switch to Aya): `Qwen/Qwen2.5-7B-Instruct`, Apache-2.0. Kept in `results/` for comparison only.

## Datasets, benchmarks and licenses

- **Source of data:** the 30 pairs of the Bridges game (`data/pairs.json`, six core pairs used here), and the answers of 27 players collected at [mapideas.org/bridges](https://mapideas.org/bridges) with consent, no names, cookies or IP addresses. Player answers are private and are not published; only the measures derived from them are.
- **Language:** English for the measures. Bridges runs in five languages; the steering is in English.
- **Train / test splits, leak check:** nothing is trained. The steering vectors are built from the model's own answers to the six core pairs; the measures compare answers to the same pairs against the human map. People are measured half against half, so they are never compared with themselves.
- **Safety and capability checks (AuditKIT):** Global-MMLU-Lite (CohereLabs, English, 120 questions), MultiJail (English, 60 harmful prompts), XSTest v2 (safe prompts that look unsafe, 60). All three are retention checks.

## Metrics, and why these

| Metric | Better | Why |
|---|---|---|
| Recall on the human map of ideas | higher | how much of the region people's answers cover the model reaches (k-nearest-neighbour spheres, all-MiniLM-L6-v2, k=2) |
| Precision on the human map | higher | how many of the model's answers land where people go, so that reaching far by scattering is not rewarded |
| Voice repetition (compression ratio of parts of speech) | lower, toward people | how much the answers repeat their sentence patterns |
| Global-MMLU-Lite accuracy | higher, should not drop | general knowledge kept |
| Refusal of harmful requests (MultiJail) | higher, should not drop | safety kept |
| Refusal of safe requests (XSTest) | lower, should not rise | no over-refusal |

The ideas are measured on content words only, with the scaffolding and the two words of the pair removed and each answer cut to a player's length, so the measure is not moved by the voice it is trying to change. No language model judges any answer.

## Method and experiments

- Layers: the middle 40% of Tiny Aya Global's decoder layers (layers 10 to 24 of 36).
- Sampling: temperature 1.0, top-p 1.0, up to 220 new tokens, 10 samples per pair and direction (120 per condition); AuditKIT checks greedy.
- Seeds: generation seed 0; the measures are repeated over 3 seeds of the measurement draws.
- Ablations: two vectors (life, voice) at three strengths each.

## Results

### Same model before and after

Tiny Aya Global, six core Bridges pairs in both directions, 120 answers per condition; recall and precision on the human map of ideas (mean over 3 measurement seeds, spread under 0.05); voice is the compression ratio of parts of speech (lower is more varied; people 4.1). AuditKIT checks are greedy, on 120 Global-MMLU-Lite questions, 60 MultiJail and 60 XSTest safe prompts, all in English.

| Condition | Recall ↑ | Precision ↑ | Voice ↓ | Global-MMLU-Lite ↑ | Refuses harmful ↑ | Refuses safe ↓ |
|---|---|---|---|---|---|---|
| Tiny Aya Base (before chat training) | 0.76 | 0.72 | 4.1 | | | |
| Tiny Aya Global, plain | 0.64 | 0.72 | 4.7 | 0.46 | 0.65 | 0.05 |
| Global, ordinary-life prompt | 0.77 | 0.61 | 4.6 | | | |
| **Global, voice vector 0.25** | **0.72** | **0.73** | 4.5 | 0.45 | 0.70 | 0.07 |
| **Global, voice vector 0.5** | **0.81** | 0.67 | 4.3 | 0.46 | 0.80 | 0.10 |
| Global, voice vector 1 | 0.64 | 0.57 | 4.3 | 0.39 | 0.83 | 0.12 |
| Global, life vector 0.25 | 0.21 | 0.46 | 4.6 | | | |
| Global, life vector 0.5 | 0.03 | 0.27 | 7.3 | 0.00 | 0.00 | 0.00 |
| Global, life vector 1 | 0.01 | 0.37 | 8.0 | 0.00 | 0.00 | 0.00 |
| People (half against half) | 0.81 | 0.82 | 4.1 | | | |

Full numbers: `results/aya/measures.csv` and `results/aya/auditkit.json`.

### Findings

**Chat training narrows Tiny Aya.** The chat model reaches less of the human map than its own base model (0.64 against 0.76), with a more repetitive voice, as Qwen2.5 did in Epistemic Bridges.

**The voice vector widens the ideas with no prompt.** At 0.25 recall rises by 8 points with precision unchanged. At 0.5 recall reaches the level of people, and the model does slightly better on both measures than when it is prompted with an ordinary life. The voice moves toward people at the same time.

**Knowledge and safety hold at the useful strengths.** At 0.25 and 0.5, Global-MMLU-Lite stays at 0.45 to 0.46, and refusal of harmful requests rises (0.65 to 0.70 and 0.80). The cost is a little more over-refusal at 0.5 (3 more of 60 safe prompts refused). At 1, knowledge starts to fall.

**A small model's idea of an ordinary person is a stereotype.** See below.

### What didn't work

The life vector, taken from the prompt with and without an ordinary life, did not widen anything. At 0.25 the model answered in a thick, invented "ordinary" dialect ("back home they made me do all this… you know like, ahhh…"), and from 0.5 it produced repeated syllables, failing every AuditKIT check. The prompt itself works (recall 0.77), so the problem is in the direction: what separates "an ordinary person is speaking" from "the assistant is speaking" in the model's activations is mostly a register, and pushing along it brings out the model's stereotype of such a person, not their ideas. The voice vector, built from how the model's own answers change, carries the ideas instead. In a pilot on Qwen2.5-7B-Instruct, the life vector did widen the ideas at 0.5 (recall 0.61 to 0.70, with precision falling to 0.65) before breaking at 1, so how far this direction carries ideas or caricature depends on the model. Steering for diversity needs to be checked against real people, since the model's picture of who counts as different is itself narrow.

### Compute used

One NVIDIA L40S on Modal, about 0.4 GPU hours for the whole run, paid from the author's own credits.

## Limitations

- One small model, six pairs, English only, 27 players: the human map is small, and the measures carry its uncertainty.
- The ordinary lives are 30 short sketches written for Bridges; a vector built from other lives may move the model elsewhere.
- Steering widens what the model reaches by changing who answers; it does not add knowledge.
- AuditKIT checks use small samples (120, 60, 60).

## With more credits

- **Planned approach:** all 30 pairs and the five languages of Bridges; Aya Expanse 8B and 32B; CircuitKIT to locate where the assistant voice lives and steer only there; a vector per kind of life instead of one average; more players to grow the human map.
- **Expected before and after:** if the light push holds across languages and sizes, recall toward the human level with precision kept, and no loss in knowledge or refusal.

## Artifacts

- **HF dataset collection:** https://huggingface.co/collections/nefinia/steering-the-voice-data-6afa430b3ce7e09c3379de6d (dataset `nefinia/steering-the-voice-tiny-aya`)
- **HF models collection (steering vectors):** https://huggingface.co/collections/nefinia/steering-the-voice-vectors-6afa430c9600a98c9d0e2800 (`nefinia/steering-the-voice-tiny-aya-vectors`)
- **GitHub repo:** this repository.

## Documentation and reproduction

- `steer_modal.py`: the whole pipeline on Modal (one L40S): answers, SafeTune vectors and steering, AuditKIT checks.
- `upload_hf.py`: publishes the answers, measures, vectors and AuditKIT log on Hugging Face.
- `data/`: the pairs and instructions of Bridges, and the ordinary lives.
- `results/aya/`: the answers, vectors and AuditKIT log of the Tiny Aya run, and `measures.csv`.
- `.github/workflows/run.yml`: runs either step from the Actions tab. Needs `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and `HF_TOKEN` (an account that accepted the Tiny Aya licence).

```
pip install modal
modal run steer_modal.py
```

The measures use the analysis code of [Epistemic Bridges](https://github.com/nefinia/epistemic-bridges) (`analysis/voice_content.py --runs aya_steer`), which needs the private player answers.

## Declaration of work done outside the hackathon

The Bridges game, its pairs and the player answers come from Epistemic Bridges, made for the Apart Research AI x Epistemics sprint (13 to 16 November 2026), and the measures are its analysis code. The steering, the Tiny Aya runs, the use of SafeTune and AuditKIT, and this write-up were made for this hackathon. A first steering pilot on Qwen2.5-7B-Instruct was also made during the hackathon, before the switch to Aya.

## Teammates

Sofia Gallego (solo), CosmoVentures, Paris.

## LLM usage statement

Claude wrote most of the code and drafted this write-up with the author; the idea, the game and the human data are the author's. No language model judges any answer.
