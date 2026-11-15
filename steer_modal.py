"""Steering the assistant voice of Tiny Aya, with SafeTune, checked with AuditKIT (Lexsi Labs).

Epistemic Bridges found that most of what makes chat answers look narrow against the answers of real
people is one shared assistant voice, and that asking a model to answer as an ordinary person drawn at
random widens its ideas. Here the same effect is sought inside the model, with no prompt.

Pipeline, all on one GPU:
1. Answers to the six core Bridges pairs, both directions, n each, from
   - Tiny Aya Base (before chat training, the text given to complete),
   - Tiny Aya Global with no system message (the assistant voice),
   - Tiny Aya Global answering as an ordinary life drawn at random (the prompted reference).
2. SafeTune: two contrastive activation addition vectors at the middle layers,
   - life: the prompt with an ordinary life as system message minus the same prompt with none (last token),
   - voice: the model's own answers as a life minus its own plain answers (mean over the answer).
3. SafeTune: steered answers from Tiny Aya Global at several strengths, with no system message.
4. AuditKIT: what should not get worse, on the plain and the steered model:
   general knowledge (Global-MMLU-Lite, English), refusal of harmful requests (MultiJail, English)
   and over-refusal of safe requests that look unsafe (XSTest, safe prompts).

The answers are measured against the human map of ideas with the Epistemic Bridges measures
(recall, precision, voice), outside Modal, because the player answers stay private.

    modal run steer_modal.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
CHAT = "CohereLabs/tiny-aya-global"
BASE = "CohereLabs/tiny-aya-base"
STRENGTHS = [0.25, 0.5, 1.0]
SEED = 0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .uv_pip_install("safetune==0.1.21", "auditkit[interop]", "transformers>=4.57", "accelerate", "hf_transfer",
                    "datasets")
    .add_local_file(str(ROOT / "data" / "pairs.json"), "/root/pairs.json")
    .add_local_file(str(ROOT / "data" / "creative.py"), "/root/creative.py")
)
app = modal.App("steering-the-voice-aya")
results = modal.Volume.from_name("steering-results", create_if_missing=True)
hf_cache = modal.Volume.from_name("bridges-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=120 * 60, secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])],
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def run(n: int = 10, audit_n: int = 60) -> str:
    import sys
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetune.interventions.steer import CAAConfig, CAAModel, extract_caa_vectors

    sys.path.insert(0, "/root")
    import creative

    out_dir = Path("/results/aya")
    out_dir.mkdir(parents=True, exist_ok=True)
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
    rows_path = out_dir / "tiny-aya_steer.jsonl"
    rows_f = rows_path.open("w")

    def write(model_id, kind, condition, meta, text, rep, extra=None):
        rows_f.write(json.dumps({"source": "ai", "model": model_id, "model_kind": kind, "year": 2026, "size_b": 3.35,
                                 "series": "aya_steer", "run": "aya_steer", "instructions": "v2", "replicate": rep,
                                 **meta, "condition": condition, "temperature": 1.0, **(extra or {}),
                                 "final": text.strip(), "raw": text}, ensure_ascii=False) + "\n")

    def load(mid):
        tok = AutoTokenizer.from_pretrained(mid)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        return tok, model

    def sample_texts(model, tok, prompts, max_new_tokens=220, reps=1, greedy=False):
        out = [[] for _ in prompts]
        for r in range(reps):
            torch.manual_seed(SEED + r)
            for i in range(0, len(prompts), 32):
                batch = prompts[i:i + 32]
                enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
                with torch.no_grad():
                    gen = model.generate(**enc, do_sample=not greedy, temperature=None if greedy else 1.0,
                                         top_p=None if greedy else 1.0, max_new_tokens=max_new_tokens,
                                         pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for j, t in enumerate(texts):
                    out[i + j].append(t)
        return out

    # 1a. the base model, before chat training: the prompt as text to complete, first paragraph kept
    tok, model = load(BASE)
    base_prompts = [f"{u}\nAnswer:" for _, u in tasks]
    for (meta, _), texts in zip(tasks, sample_texts(model, tok, base_prompts, reps=n)):
        for rep, t in enumerate(texts, 1):
            write(BASE, "base", "baseline", meta, t.strip().split("\n\n")[0], rep, {"prompt_mode": "completion"})
    del model
    torch.cuda.empty_cache()
    print("base: done")

    # 1b. the chat model, plain and as an ordinary life
    tok, model = load(CHAT)
    n_layers = model.config.num_hidden_layers
    layers = list(range(int(n_layers * 0.3), int(n_layers * 0.7)))

    def chat(user, system=None, answer=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text + answer if answer is not None else text

    rng = random.Random(SEED)
    plain_prompts = [chat(u) for _, u in tasks]
    plain = sample_texts(model, tok, plain_prompts, reps=n)
    life_systems = [[f"You are {rng.choice(creative.LIVES)}. Answer as yourself, in your own words." for _ in range(n)]
                    for _ in tasks]
    life_answers = [[] for _ in tasks]
    for r in range(n):
        got = sample_texts(model, tok, [chat(u, life_systems[i][r]) for i, (_, u) in enumerate(tasks)])
        for i, g in enumerate(got):
            life_answers[i].append(g[0])
    for i, (meta, _) in enumerate(tasks):
        for rep in range(n):
            write(CHAT, "chat", "baseline", meta, plain[i][rep], rep + 1, {"steer_vector": "none", "steer_strength": 0.0})
            write(CHAT, "chat", "ordinary_life", meta, life_answers[i][rep], rep + 1, {"system": life_systems[i][rep]})
    rows_f.flush()
    print("chat: done")

    # 2. SafeTune: the two steering vectors
    life_pos = [chat(u, life_systems[i][r]) for i, (_, u) in enumerate(tasks) for r in range(n)]
    life_neg = [chat(u) for _, u in tasks for _ in range(n)]
    voice_pos = [chat(u, answer=life_answers[i][r]) for i, (_, u) in enumerate(tasks) for r in range(n)]
    voice_neg = [chat(u, answer=plain[i][r]) for i, (_, u) in enumerate(tasks) for r in range(n)]
    vectors = {
        "life": extract_caa_vectors(model, tok, life_pos, life_neg, CAAConfig(target_layers=layers, pool_method="last_token")),
        "voice": extract_caa_vectors(model, tok, voice_pos, voice_neg, CAAConfig(target_layers=layers, pool_method="mean")),
    }
    norms = {name: {int(l): float(v.norm()) for l, v in vec.items()} for name, vec in vectors.items()}
    torch.save({name: {int(l): v.detach().cpu() for l, v in vec.items()} for name, vec in vectors.items()},
               out_dir / "steering_vectors.pt")
    (out_dir / "steering_vectors.json").write_text(json.dumps({"model": CHAT, "layers": layers, "n_layers": n_layers,
                                                               "norms": norms}, indent=1))
    print("vectors:", json.dumps({k: round(sum(v.values()) / len(v), 2) for k, v in norms.items()}))

    # 3. SafeTune: steered answers, no system message
    wrappers = {"none": model}
    for name in vectors:
        for s in STRENGTHS:
            wrapper = CAAModel(model, vectors[name], strength=s)
            got = sample_texts(wrapper, tok, plain_prompts, reps=n)
            for (meta, _), texts in zip(tasks, got):
                for rep, t in enumerate(texts, 1):
                    write(CHAT, "chat", f"{name}_{s:g}", meta, t, rep, {"steer_vector": name, "steer_strength": s,
                                                                          "steer_layers": layers})
            rows_f.flush()
            wrappers[f"{name}_{s:g}"] = wrapper
            print(f"{name}_{s:g}: done")
    rows_f.close()

    # 4. AuditKIT: what should not get worse
    import auditkit as ak
    from auditkit.benchmark_loaders import GLOBAL_MMLU_LITE, load_global_mmlu, load_multijail, load_xstest
    from auditkit.metrics.refusal import find_refusal

    sets = {}
    for name, fn in (("global_mmlu_lite_en", lambda: load_global_mmlu("en", repo=GLOBAL_MMLU_LITE, limit=audit_n * 2)),
                     ("multijail_en", lambda: load_multijail("en", limit=audit_n)),
                     ("xstest_safe", lambda: [s for s in load_xstest() if "safe" in s.tags][:audit_n])):
        try:
            sets[name] = fn()
        except Exception as e:  # a gated or moved dataset should not stop the rest
            print(f"auditkit: could not load {name}: {e}")

    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def mcq_correct(sample, output):
        m = re.search(r"\d", output or "")
        return 1.0 if m and int(m.group()) == sample.target else 0.0

    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def refused(sample, output):
        return 1.0 if find_refusal(output or "", "en") else 0.0

    audit = {}
    for cond in ["none", "life_0.5", "voice_0.25", "life_1", "voice_1"]:
        if cond not in wrappers:
            continue
        wrapper = wrappers[cond]

        def as_model(prompts, wrapper=wrapper, cond=cond):
            short = all(p.rstrip().endswith("Answer:") for p in prompts)
            got = sample_texts(wrapper, tok, [chat(p) for p in prompts], max_new_tokens=6 if short else 96, greedy=True)
            return [g[0] for g in got]

        audit[cond] = {}
        for name, samples in sets.items():
            try:
                res = ak.evaluate(samples, as_model, [mcq_correct] if name.startswith("global") else [refused],
                                  adapter="mcq" if name.startswith("global") else None, cache=False)
                scores = [float(v) for v in _per_sample(res)]
                audit[cond][name] = {"mean": sum(scores) / max(len(scores), 1), "n": len(scores),
                                     "summary": str(res.summary())[:2000]}
            except Exception as e:
                audit[cond][name] = {"error": repr(e)[:500]}
            print(cond, name, audit[cond][name].get("mean", audit[cond][name].get("error")))
    (out_dir / "auditkit.json").write_text(json.dumps(audit, indent=1))
    results.commit()
    return f"{CHAT}: answers, vectors and audit written"


def _per_sample(res):
    """The per-sample scores of an AuditKIT RunResult, whatever its exact shape in this version."""
    for attr in ("sample_results", "samples", "results", "rows"):
        items = getattr(res, attr, None)
        if items:
            vals = []
            for it in items:
                scores = getattr(it, "scores", None) or (it.get("scores") if isinstance(it, dict) else None)
                if isinstance(scores, dict):
                    v = next(iter(scores.values()))
                elif isinstance(scores, list) and scores:
                    v = scores[0]
                else:
                    v = getattr(it, "score", None)
                v = getattr(v, "value", v)
                if v is not None:
                    vals.append(v)
            if vals:
                return vals
    raise ValueError(f"no per-sample scores found on {type(res).__name__}: {[a for a in dir(res) if not a.startswith('_')]}")


@app.function(image=image, gpu="L40S", timeout=60 * 60, secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])],
              volumes={"/results": results, "/root/.cache/huggingface": hf_cache})
def audit_only(conds: str = "none,voice_0.5", audit_n: int = 60) -> str:
    """AuditKIT checks for extra conditions, with the vectors saved by run()."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetune.interventions.steer import CAAModel
    import auditkit as ak
    from auditkit.benchmark_loaders import GLOBAL_MMLU_LITE, load_global_mmlu, load_multijail, load_xstest
    from auditkit.metrics.refusal import find_refusal

    out_dir = Path("/results/aya")
    vecs = torch.load(out_dir / "steering_vectors.pt")
    tok = AutoTokenizer.from_pretrained(CHAT)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(CHAT, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    vecs = {k: {l: v.to("cuda", dtype=torch.bfloat16) for l, v in d.items()} for k, d in vecs.items()}

    def chat(user):
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

    def gen(m, prompts, max_new_tokens):
        out = []
        for i in range(0, len(prompts), 32):
            enc = tok(prompts[i:i + 32], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                g = m.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            out += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out

    sets = {"global_mmlu_lite_en": load_global_mmlu("en", repo=GLOBAL_MMLU_LITE, limit=audit_n * 2),
            "multijail_en": load_multijail("en", limit=audit_n),
            "xstest_safe": [s for s in load_xstest() if "safe" in s.tags][:audit_n]}

    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def mcq_correct(sample, output):
        m = re.search(r"\d", output or "")
        return 1.0 if m and int(m.group()) == sample.target else 0.0

    @ak.scorer(direction=ak.Direction.MAXIMIZE)
    def refused(sample, output):
        return 1.0 if find_refusal(output or "", "en") else 0.0

    path = out_dir / "auditkit.json"
    audit = json.loads(path.read_text()) if path.exists() else {}
    for cond in conds.split(","):
        if cond == "none":
            m = model
        else:
            name, s = cond.rsplit("_", 1)
            m = CAAModel(model, vecs[name], strength=float(s))

        def as_model(prompts, m=m):
            short = all(p.rstrip().endswith("Answer:") for p in prompts)
            return gen(m, [chat(p) for p in prompts], 6 if short else 96)

        audit[cond] = {}
        for name, samples in sets.items():
            res = ak.evaluate(samples, as_model, [mcq_correct] if name.startswith("global") else [refused],
                              adapter="mcq" if name.startswith("global") else None, cache=False)
            scores = [float(v) for v in _per_sample(res)]
            audit[cond][name] = {"mean": sum(scores) / max(len(scores), 1), "n": len(scores),
                                 "summary": str(res.summary())[:2000]}
            print(cond, name, audit[cond][name]["mean"])
    path.write_text(json.dumps(audit, indent=1))
    results.commit()
    return "audit written"


@app.local_entrypoint()
def main(n: int = 10, audit_n: int = 60, audit: str = ""):
    print(audit_only.remote(audit, audit_n) if audit else run.remote(n, audit_n))
