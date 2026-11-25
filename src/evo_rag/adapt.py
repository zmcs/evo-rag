"""Week 4, step 2: adapt bge-small to the AI Act in four ways (free: runs on your computer).

All methods learn from the same synthetic (question, passage) pairs (synth_data.py) and
retrieve from the same passage pool (every passage of the reference chunking).

  diag-grad    one weight per embedding dimension (384), gradient descent on a contrastive loss
  diag-cmaes   the SAME 384 weights, found by CMA-ES maximising the ranking metric (MRR) directly
  linear       a full 384x384 query matrix, gradient descent on the contrastive loss
  lora         LoRA adapters inside bge-small (attention query/value), contrastive loss

diag-grad vs diag-cmaes isolates the optimiser (gradients vs evolution) at equal capacity;
linear and lora show what more capacity buys. For each method the best version on the
synthetic validation split is kept, then saved to data/models/bge-small-<method>.

  uv run python -m evo_rag.adapt                        # all four
  uv run python -m evo_rag.adapt --methods diag-cmaes   # just one
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from evo_rag.chunking import chunk_corpus
from evo_rag.retrieval import EMBEDDERS, MODELS_DIR, get_embedder, load_corpus
from evo_rag.synth_data import OUT_PATH as PAIRS_PATH, REFERENCE_CHUNKING

RESULTS_DIR = Path("data/results")
BASE = "bge-small"
SCALE = 20.0  # temperature of the contrastive loss (standard for sentence-transformers)


def ranks(queries: np.ndarray, pool: np.ndarray, positives: np.ndarray) -> np.ndarray:
    """1-based rank of each query's positive passage among the whole pool."""
    scores = queries @ pool.T
    return 1 + (scores > scores[np.arange(len(positives)), positives][:, None]).sum(axis=1)


def metrics(queries: np.ndarray, pool: np.ndarray, positives: np.ndarray) -> dict:
    r = ranks(queries, pool, positives)
    return {"mrr": float(np.mean(1 / r)), "recall@5": float(np.mean(r <= 5)), "recall@10": float(np.mean(r <= 10))}


def normalise(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# --- query-side adapters, gradient descent -----------------------------------------------------

def train_query_adapter(kind, q_train, p_train_pos, pool, q_val, pos_val, steps=400, seed=0):
    """Full-softmax contrastive loss over the whole pool; keeps the best step on validation."""
    import torch

    torch.manual_seed(seed)
    d = q_train.shape[1]
    Q, P = torch.tensor(q_train, dtype=torch.float32), torch.tensor(pool, dtype=torch.float32)
    target = torch.tensor(p_train_pos)
    if kind == "diag":
        w = torch.ones(d, requires_grad=True)
        lr, apply = 1e-2, lambda q: q * w
    else:
        w = torch.eye(d).clone().requires_grad_(True)
        lr, apply = 1e-3, lambda q: q @ w.T
    opt = torch.optim.Adam([w], lr=lr)
    best = (-1.0, w.detach().clone())
    for step in range(steps):
        logits = SCALE * torch.nn.functional.normalize(apply(Q), dim=1) @ P.T
        loss = torch.nn.functional.cross_entropy(logits, target)
        if kind == "linear":  # stay close to the identity: little data, many parameters
            loss = loss + 1e-2 * ((w - torch.eye(d)) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 9:
            weights = w.detach().numpy()
            adapted = q_val * weights if kind == "diag" else q_val @ weights.T
            val = metrics(normalise(adapted), pool, pos_val)["mrr"]
            if val > best[0]:
                best = (val, w.detach().clone())
    return best[1].numpy()


# --- query-side adapter, CMA-ES -----------------------------------------------------------------

def train_diag_cmaes(q_train, pos_train, pool, q_val, pos_val, max_evals=6000, seed=0):
    """Separable CMA-ES over the 384 per-dimension weights, maximising training MRR directly."""
    import cma

    def mrr(weights, q, pos):
        return metrics(normalise(q * weights), pool, pos)["mrr"]

    es = cma.CMAEvolutionStrategy(
        np.ones(q_train.shape[1]), 0.1,
        {"CMA_diagonal": True, "maxfevals": max_evals, "seed": seed + 1, "verbose": -9},
    )
    best = (-1.0, np.ones(q_train.shape[1]))
    while not es.stop():
        candidates = es.ask()
        es.tell(candidates, [-mrr(np.asarray(c), q_train, pos_train) for c in candidates])
        val = mrr(np.asarray(es.result.xfavorite), q_val, pos_val)
        if val > best[0]:
            best = (val, np.asarray(es.result.xfavorite).copy())
    return best[1], es.result.evaluations


# --- LoRA inside the model ----------------------------------------------------------------------

def train_lora(pairs_train, val_queries, val_pos, pool_texts, epochs=2, batch=16, seed=0, base_path=None):
    """In-batch-negatives contrastive training of LoRA adapters; merged into the model afterwards."""
    import torch
    from peft import LoraConfig, get_peft_model
    from sentence_transformers import SentenceTransformer

    model_name, q_prefix, p_prefix = EMBEDDERS[BASE]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(base_path or model_name, device=device)
    # get_peft_model injects the LoRA layers into the transformer in place and freezes the rest;
    # keep a handle on the wrapper for saving, restoring and merging.
    peft_model = get_peft_model(model[0].auto_model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                                                                target_modules=["query", "value"]))
    params = [p for p in model.parameters() if p.requires_grad]
    lora_state = lambda: {k: v.detach().clone() for k, v in peft_model.state_dict().items() if "lora" in k}  # noqa: E731
    opt = torch.optim.AdamW(params, lr=2e-4)
    rng = random.Random(seed)

    # sentence-transformers renamed tokenize() to preprocess() in recent versions.
    prepare = getattr(model, "preprocess", None) or model.tokenize

    def embed(texts):
        features = prepare(texts)
        features = {k: v.to(device) for k, v in features.items() if hasattr(v, "to")}
        return torch.nn.functional.normalize(model(features)["sentence_embedding"], dim=1)

    def val_mrr():
        model.eval()
        with torch.no_grad():
            pool = np.concatenate([embed([p_prefix + t for t in pool_texts[i:i + 64]]).cpu().numpy()
                                   for i in range(0, len(pool_texts), 64)])
            q = np.concatenate([embed([q_prefix + t for t in val_queries[i:i + 64]]).cpu().numpy()
                                for i in range(0, len(val_queries), 64)])
        model.train()
        return metrics(q, pool, val_pos)["mrr"]

    best, n_trainable = val_mrr(), sum(p.numel() for p in params)
    best_state = lora_state()
    model.train()
    for _ in range(epochs):
        order = pairs_train[:]
        rng.shuffle(order)
        batches, current, seen = [], [], set()
        for pair in order:  # no passage twice in a batch: it would be a false negative
            if pair["chunk_id"] in seen or len(current) == batch:
                batches.append(current)
                current, seen = [], set()
            current.append(pair)
            seen.add(pair["chunk_id"])
        batches.append(current)
        for b in (b for b in batches if len(b) > 1):
            q = embed([q_prefix + p["query"] for p in b])
            p = embed([p_prefix + p["passage"] for p in b])
            loss = torch.nn.functional.cross_entropy(SCALE * q @ p.T, torch.arange(len(b), device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = val_mrr()
        if score > best:
            best = score
            best_state = lora_state()
    peft_model.load_state_dict(best_state, strict=False)
    peft_model.merge_and_unload()  # writes the LoRA weights into the transformer, in place
    model.eval()
    return model, n_trainable


def main(base_path: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Adapt bge-small to the AI Act.")
    parser.add_argument("--methods", default="diag-grad,diag-cmaes,linear,lora")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",")]

    if not PAIRS_PATH.exists():
        raise SystemExit(f"{PAIRS_PATH} not found: run `python -m evo_rag.synth_data --yes` first.")
    pairs = [json.loads(line) for line in PAIRS_PATH.read_text(encoding="utf-8").splitlines()]
    train, val = [p for p in pairs if p["split"] == "train"], [p for p in pairs if p["split"] == "val"]

    pool_chunks = chunk_corpus(load_corpus(), REFERENCE_CHUNKING)
    index = {c.chunk_id: i for i, c in enumerate(pool_chunks)}
    pool_texts = [c.text for c in pool_chunks]
    pos_train = np.array([index[p["chunk_id"]] for p in train])
    pos_val = np.array([index[p["chunk_id"]] for p in val])
    print(f"{len(train)} training pairs, {len(val)} validation pairs, pool of {len(pool_texts)} passages\n")

    base = get_embedder(BASE)
    pool = base.passages(pool_texts)
    q_train, q_val = base.queries([p["query"] for p in train]), base.queries([p["query"] for p in val])

    results = {"baseline": {"params": 0, "seconds": 0.0, **metrics(q_val, pool, pos_val)}}
    print(f"{'method':12} {'params':>8} {'val MRR':>8} {'R@5':>6} {'R@10':>6} {'time':>7}")
    print(f"{'baseline':12} {0:8d} {results['baseline']['mrr']:8.3f} {results['baseline']['recall@5']:6.3f} "
          f"{results['baseline']['recall@10']:6.3f} {'-':>7}", flush=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for method in methods:
        start = time.perf_counter()
        folder = MODELS_DIR / f"{BASE}-{method}"
        extra = {}
        if method in ("diag-grad", "linear"):
            kind = "diag" if method == "diag-grad" else "linear"
            w = train_query_adapter(kind, q_train, pos_train, pool, q_val, pos_val, seed=args.seed)
            adapted_val = normalise(q_val * w if kind == "diag" else q_val @ w.T)
            n_params = w.size
        elif method == "diag-cmaes":
            w, evals = train_diag_cmaes(q_train, pos_train, pool, q_val, pos_val, seed=args.seed)
            adapted_val, n_params, extra = normalise(q_val * w), w.size, {"evaluations": int(evals)}
        elif method == "lora":
            model, n_params = train_lora(train, [p["query"] for p in val], pos_val, pool_texts,
                                         seed=args.seed, base_path=base_path)
            folder.mkdir(parents=True, exist_ok=True)
            model.save(str(folder))
            _, q_prefix, p_prefix = EMBEDDERS[BASE]
            encode = lambda texts: model.encode(texts, batch_size=32, normalize_embeddings=True)  # noqa: E731
            m = metrics(encode([q_prefix + p["query"] for p in val]), encode([p_prefix + t for t in pool_texts]), pos_val)
            results[method] = {"params": int(n_params), "seconds": time.perf_counter() - start, **m}
        else:
            raise SystemExit(f"Unknown method {method}")
        if method != "lora":
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "adapter.npy", w)
            results[method] = {"params": int(n_params), "seconds": time.perf_counter() - start,
                               **metrics(adapted_val, pool, pos_val), **extra}
        r = results[method]
        print(f"{method:12} {r['params']:8d} {r['mrr']:8.3f} {r['recall@5']:6.3f} {r['recall@10']:6.3f} "
              f"{r['seconds']:6.0f}s", flush=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"adapters_{stamp}.json").write_text(json.dumps(results, indent=1))
    print(f"\nSaved adapters to {MODELS_DIR}/ and results to data/results/adapters_{stamp}.json")
    print("Validation here is on synthetic questions. Next: `python -m evo_rag.adapt_check` for the real QA set.")


if __name__ == "__main__":
    main()
