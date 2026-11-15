"""Publish the artifacts on Hugging Face, public, in two collections (data, and steering vectors).

Needs HF_TOKEN with write rights. Player answers are never uploaded: only AI answers, measures,
vectors and AuditKIT logs.

    python upload_hf.py
"""

import os
from pathlib import Path

from huggingface_hub import HfApi

USER = "nefinia"
DATA = f"{USER}/steering-the-voice-tiny-aya"
VECS = f"{USER}/steering-the-voice-tiny-aya-vectors"
ROOT = Path(__file__).resolve().parent / "results" / "aya"

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(DATA, repo_type="dataset", private=False, exist_ok=True)
api.create_repo(VECS, repo_type="model", private=False, exist_ok=True)
for f in ["tiny-aya_steer.jsonl", "auditkit.json", "measures.csv", "steering_vectors.json"]:
    if (ROOT / f).exists():
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f, repo_id=DATA, repo_type="dataset")
api.upload_file(path_or_fileobj=str(Path(__file__).parent / "hf" / "dataset_card.md"), path_in_repo="README.md",
                repo_id=DATA, repo_type="dataset")
for f in ["steering_vectors.pt", "steering_vectors.json"]:
    if (ROOT / f).exists():
        api.upload_file(path_or_fileobj=str(ROOT / f), path_in_repo=f, repo_id=VECS, repo_type="model")
api.upload_file(path_or_fileobj=str(Path(__file__).parent / "hf" / "model_card.md"), path_in_repo="README.md",
                repo_id=VECS, repo_type="model")

for title, repo, kind in (("Steering the voice: data", DATA, "dataset"), ("Steering the voice: vectors", VECS, "model")):
    col = api.create_collection(title=title, namespace=USER, private=False, exists_ok=True,
                                description="Lexsi AI Safety and Alignment Hackathon, Tiny Aya Global steered toward "
                                            "the range of ideas real people reach.")
    try:
        api.add_collection_item(col.slug, item_id=repo, item_type=kind, exists_ok=True)
    except Exception as e:
        print(e)
    print(title, f"https://huggingface.co/collections/{col.slug}")
