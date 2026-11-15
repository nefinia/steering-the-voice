---
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
tags: [steering, contrastive-activation-addition, safetune, tiny-aya]
---
# Steering vectors for Tiny Aya Global

Two contrastive activation addition vectors made with SafeTune for the project
[steering-the-voice](https://github.com/nefinia/steering-the-voice), at the middle layers of
`CohereLabs/tiny-aya-global`:

- `life`: the prompt with an ordinary life as system message, minus the same prompt with none (last token).
- `voice`: the model's own answers as an ordinary life, minus its plain answers (mean over the answer).

`steering_vectors.pt` holds `{name: {layer: tensor}}`; load it and pass a vector to SafeTune's `CAAModel`
with a strength (0.25 to 0.5 widened the ideas; 1 and above started to break the answers).
