"""Steering the assistant voice out of a chat model, with SafeTune (Lexsi Labs).

Bridges found that most of what makes chat answers look narrow is one shared assistant voice, and that
asking a model to answer as an ordinary person drawn at random widens its ideas. Here the same effect is
sought inside the model, with no prompt: contrastive activation addition (CAA, Panickssery et al. 2023,
as implemented in SafeTune's steer module) on Qwen2.5-7B-Instruct, the model with the clearest
before-and-after in Bridges.

Two steering vectors, each the mean activation of one set minus another, at the middle layers:
- life: the Bridges prompt with an ordinary-life system message, minus the same prompt with none
  (last prompt token, so the vector holds "who is answering" before any word is written);
- voice: the model's own answers given as an ordinary life, minus its own answers with no persona,
  read in full after the same prompt (mean over tokens, so the vector holds how the answer is written).

Each vector is added at several strengths while the model answers the six core pairs, in both
directions, ten times each, with no system message. The answers go to the same volume as the other
runs (run "steer") and are measured with analysis/voice_content.py like every other condition.

    modal run steer_modal.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = list(range(8, 20))            # middle layers of 28
STRENGTHS = [0.25, 0.5, 1.0, 2.0]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .uv_pip_install("safetune==0.1.21", "hf_transfer")
    .add_local_file(str(ROOT / "data" / "pairs.json"), "/root/pairs.json")
    .add_local_file(str(ROOT / "data" / "creative.py"), "/root/creative.py")
    .add_local_file(str(ROOT / "data" / "qwen2.5-7b-instruct_creative.jsonl"), "/root/creative_run.jsonl")
)
app = modal.App("steering-the-voice")
results = modal.Volume.from_name("steering-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("bridges-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=90 * 60,
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def run(n: int = 10) -> str:
    import sys
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetune.interventions.steer import CAAConfig, CAAModel, extract_caa_vectors

    sys.path.insert(0, "/root")
    import creative

    pairs = json.loads(Path("/root/pairs.json").read_text())
    core = [p for p in pairs["pairs"] if p["set"] == "core"]
    welcome = pairs["instructions_versions"]["v2"]["welcome"]["en"]
    question = pairs["instructions"]["en"]["question"]
    tasks = []
    for p in core:
        for d in ("ab", "ba"):
            src, dst = (p["a"]["en"], p["b"]["en"]) if d == "ab" else (p["b"]["en"], p["a"]["en"])
            tasks.append(({"task": "bridge", "pair": p["id"], "level": p["level"], "set": p["set"], "dir": d,
                           "lang": "en", "from": src, "to": dst}, f"{welcome}\n\n{src} → {dst}\n\n{question}"))

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def chat(user, system=None, answer=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + answer if answer is not None else text

    rng = random.Random(0)
    # life: the same prompts, with and without an ordinary life as the system message
    life_pos = [chat(u, f"You are {rng.choice(creative.LIVES)}. Answer as yourself, in your own words.") for _, u in tasks for _ in range(10)]
    life_neg = [chat(u) for _, u in tasks for _ in range(10)]
    # voice: the model's own answers as an ordinary life against its own answers with no persona,
    # both read after the plain prompt (no system message)
    rows = [json.loads(l) for l in Path("/root/creative_run.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["task"] == "bridge" and r["lang"] == "en" and r["final"].strip()]
    prompt_of = {(m["pair"], m["dir"]): u for m, u in tasks}
    voice_pos = [chat(prompt_of[(r["pair"], r["dir"])], answer=r["final"]) for r in rows if r["condition"] == "ordinary_life"]
    voice_neg = [chat(prompt_of[(r["pair"], r["dir"])], answer=r["final"]) for r in rows if r["condition"] == "baseline"]
    k = min(len(voice_pos), len(voice_neg), 120)
    voice_pos, voice_neg = voice_pos[:k], voice_neg[:k]

    vectors = {
        "life": extract_caa_vectors(model, tok, life_pos, life_neg, CAAConfig(target_layers=LAYERS, pool_method="last_token")),
        "voice": extract_caa_vectors(model, tok, voice_pos, voice_neg, CAAConfig(target_layers=LAYERS, pool_method="mean")),
    }
    norms = {name: {int(l): float(v.norm()) for l, v in vec.items()} for name, vec in vectors.items()}
    print(json.dumps(norms))

    users = [u for _, u in tasks]
    enc = tok([chat(u) for u in users], return_tensors="pt", padding=True).to("cuda")

    def generate(wrapper=None):
        out_texts = []
        for _ in range(n):
            with torch.no_grad():
                gen = (wrapper or model).generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=220,
                                                  pad_token_id=tok.eos_token_id)
            out_texts.append(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return out_texts

    conditions = [("none", 0.0, None)]
    for name in vectors:
        for s in STRENGTHS:
            conditions.append((name, s, CAAModel(model, vectors[name], strength=s)))

    path = Path(f"/results/{MODEL.replace('/', '__')}__steer.jsonl")
    count = 0
    with path.open("w") as f:
        for name, s, wrapper in conditions:
            torch.manual_seed(0)
            samples = generate(wrapper)
            cond = "baseline" if name == "none" else f"{name}_{s:g}"
            for rep, texts in enumerate(samples, start=1):
                for (meta, _), text in zip(tasks, texts):
                    f.write(json.dumps({"source": "ai", "model": MODEL, "year": 2024, "size_b": 7, "model_kind": "chat",
                                        "series": "steer", "prompt_mode": "chat", "run": "steer", "instructions": "v2",
                                        "replicate": rep, **meta, "condition": cond, "temperature": 1.0,
                                        "steer_vector": name, "steer_strength": s, "steer_layers": LAYERS,
                                        "final": text.strip(), "raw": text}, ensure_ascii=False) + "\n")
                    count += 1
            print(f"{cond}: done")
    (Path("/results") / "steer_vector_norms.json").write_text(json.dumps(norms))
    results.commit()
    return f"{MODEL}: {count} steered answers"


@app.local_entrypoint()
def main(n: int = 10):
    print(run.remote(n))
