#!/usr/bin/env python3
"""
Bangladesh Legal QA — VS Code-safe retrieval/reranking pipeline.

Current architecture:
    Question
      -> BM25 Top-50
      -> Hard-negative multilingual E5 Top-50
      -> deduplicated union (~95 candidates)
      -> BAAI/bge-reranker-v2-m3
      -> Top-10 retrieval / Top-5 RAG evidence

Design goals:
- no notebook/browser output rendering
- compact terminal logs
- automatic checkpoints
- resume after interruption
- 300-DPI report figures saved to files
- citation-aware train/validation/test split
- validation for tuning; final test only after the model is frozen
- no bge-reranker-v2-gemma (too memory-heavy for the earlier T4 run)

Known validation reference from the current experiment:
BM25 Top-50: 55.51%
Hard-negative E5 Top-50: 73.06%
BM25+E5 union: 85.31%
BGE v2-m3 Recall@1/3/5/10: 45.71/57.96/65.31/74.29%
MRR@10: 0.5403
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    work_dir: str = "./artifacts"
    input_json: Optional[str] = None
    split_dir: Optional[str] = None
    hardneg_model: Optional[str] = None

    hf_repo: str = "momahadi/bangladesh-legal-qa-dataset"
    hf_file: str = "sft/finetune_dataset_2165.json"

    seed: int = 42
    device: str = "auto"

    dense_top_k: int = 50
    bm25_top_k: int = 50
    reranker_top_k: int = 10

    e5_batch_size: int = 32
    train_batch_size: int = 16
    stage1_epochs: int = 4
    stage2_epochs: int = 2

    e5_base_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512

    resume: bool = True
    final_test: bool = False


@dataclass
class Paths:
    root: Path
    data: Path
    splits: Path
    models: Path
    checkpoints: Path
    results: Path
    figures: Path
    logs: Path
    cache: Path

    @classmethod
    def make(cls, work_dir: str) -> "Paths":
        root = Path(work_dir).resolve()
        obj = cls(
            root=root,
            data=root/"data",
            splits=root/"splits",
            models=root/"models",
            checkpoints=root/"checkpoints",
            results=root/"results",
            figures=root/"figures",
            logs=root/"logs",
            cache=root/"cache",
        )
        for p in asdict(obj).values():
            Path(p).mkdir(parents=True, exist_ok=True)
        return obj


# ---------------------------------------------------------------------------
# Logging / filesystem / memory
# ---------------------------------------------------------------------------

def get_logger(paths: Paths) -> logging.Logger:
    logger = logging.getLogger("legalqa")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(paths.logs/"pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return value


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: Path) -> None:
    def conv(x):
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        return x

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(conv(obj), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Dataset loading / cleaning
# ---------------------------------------------------------------------------

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return re.sub(r"\s+", " ", str(x).replace("\u00a0", " ")).strip()


def normalize_act(x: Any) -> str:
    return clean_text(x).lower().strip(" ,.;:-")


def normalize_section(x: Any) -> str:
    s = clean_text(x).lower()
    s = s.replace("section", "").replace("sec.", "").replace("sec", "")
    return re.sub(r"\s+", " ", s).strip(" ,.;:-")


def canonical_citation(act: str, section: str, subsection: str = "") -> str:
    parts = []
    if act:
        parts.append(act)
    if section:
        parts.append(f"section {section}")
    if subsection:
        parts.append(subsection)
    return " | ".join(parts)


def dataset_path(cfg: Config, paths: Paths, logger: logging.Logger) -> Path:
    if cfg.input_json:
        p = Path(cfg.input_json).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    target = paths.data/"finetune_dataset_2165.json"
    if target.exists():
        return target

    logger.info("Downloading dataset from Hugging Face...")
    from huggingface_hub import hf_hub_download

    src = hf_hub_download(
        repo_id=cfg.hf_repo,
        filename=cfg.hf_file,
        repo_type="dataset",
    )
    shutil.copy2(src, target)
    return target


def load_raw(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            payload = payload["data"]
        elif isinstance(payload.get("records"), list):
            payload = payload["records"]
        elif all(isinstance(v, dict) for v in payload.values()):
            payload = list(payload.values())
    if not isinstance(payload, list):
        raise ValueError("Dataset JSON must contain a list of records.")
    return pd.DataFrame(payload)


def prepare_df(raw: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = raw.copy()

    required = ["Question", "Section Text", "Act", "Section Number"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "dataset_id" not in df.columns:
        df["dataset_id"] = np.arange(len(df))

    df["question_clean"] = df["Question"].map(clean_text)
    df["answer_clean"] = (
        df["Answer"].map(clean_text)
        if "Answer" in df.columns else ""
    )
    df["section_text_clean"] = df["Section Text"].map(clean_text)
    df["act_normalized"] = df["Act"].map(normalize_act)
    df["section_number_normalized"] = df["Section Number"].map(normalize_section)

    if "Subsection/Clause" in df.columns:
        df["subsection_normalized"] = df["Subsection/Clause"].map(
            lambda x: clean_text(x).lower()
        )
    else:
        df["subsection_normalized"] = ""

    df["canonical_citation"] = [
        canonical_citation(a, s, sub)
        for a, s, sub in zip(
            df["act_normalized"],
            df["section_number_normalized"],
            df["subsection_normalized"],
        )
    ]

    if "language" not in df.columns:
        df["language"] = "Unknown"
    if "question_type" not in df.columns:
        df["question_type"] = "unknown"

    allowed = {"single_hop", "advanced_selection"}
    if df["question_type"].isin(allowed).any():
        df = df[df["question_type"].isin(allowed)].copy()

    df = df[
        (df["question_clean"].str.len() > 0)
        & (df["section_text_clean"].str.len() > 0)
        & (df["canonical_citation"].str.len() > 0)
    ].reset_index(drop=True)

    logger.info(
        "Prepared QA records=%d | unique citations=%d",
        len(df), df["canonical_citation"].nunique()
    )
    return df


# ---------------------------------------------------------------------------
# Citation-aware split
# ---------------------------------------------------------------------------

def make_split(df: pd.DataFrame, cfg: Config, logger: logging.Logger):
    df = df.copy()
    df["split_group_id"] = df["canonical_citation"]

    g1 = GroupShuffleSplit(
        n_splits=1, train_size=0.70, random_state=cfg.seed
    )
    train_idx, temp_idx = next(g1.split(df, groups=df["split_group_id"]))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    g2 = GroupShuffleSplit(
        n_splits=1, train_size=0.50, random_state=cfg.seed + 1
    )
    val_idx, test_idx = next(
        g2.split(temp_df, groups=temp_df["split_group_id"])
    )

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    for x, name in [(train_df, "train"), (val_df, "validation"), (test_df, "test")]:
        x["split"] = name
        x.reset_index(drop=True, inplace=True)

    tc = set(train_df["canonical_citation"])
    vc = set(val_df["canonical_citation"])
    sc = set(test_df["canonical_citation"])

    if tc & vc or tc & sc or vc & sc:
        raise RuntimeError("Citation leakage detected.")

    logger.info(
        "Split | train=%d val=%d test=%d | overlaps=0",
        len(train_df), len(val_df), len(test_df)
    )

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, split_dir: Path):
    train_df.to_json(split_dir/"train.jsonl", orient="records", lines=True, force_ascii=False)
    val_df.to_json(split_dir/"validation.jsonl", orient="records", lines=True, force_ascii=False)
    test_df.to_json(split_dir/"test.jsonl", orient="records", lines=True, force_ascii=False)


def load_splits(split_dir: Path):
    return (
        pd.read_json(split_dir/"train.jsonl", lines=True),
        pd.read_json(split_dir/"validation.jsonl", lines=True),
        pd.read_json(split_dir/"test.jsonl", lines=True),
    )


# ---------------------------------------------------------------------------
# Retrieval corpus
# ---------------------------------------------------------------------------

def build_corpus(train_df, val_df, test_df, logger):
    all_qa = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Critical: dedupe by citation + passage, NOT citation only.
    corpus = (
        all_qa[["canonical_citation", "section_text_clean"]]
        .drop_duplicates(["canonical_citation", "section_text_clean"])
        .reset_index(drop=True)
    )

    corpus["retrieval_text"] = (
        corpus["canonical_citation"].fillna("")
        + " | "
        + corpus["section_text_clean"].fillna("")
    )
    corpus["passage_id"] = np.arange(len(corpus))

    logger.info(
        "Corpus passages=%d | unique citations=%d",
        len(corpus), corpus["canonical_citation"].nunique()
    )
    return corpus


# ---------------------------------------------------------------------------
# E5 training
# ---------------------------------------------------------------------------

def train_e5_if_needed(
    cfg: Config, paths: Paths, train_df: pd.DataFrame,
    corpus: pd.DataFrame, device: str, logger: logging.Logger
) -> Path:

    if cfg.hardneg_model:
        p = Path(cfg.hardneg_model).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        logger.info("Using existing hard-negative E5: %s", p)
        return p

    final_dir = paths.models/"legal_e5_hardneg"
    if cfg.resume and final_dir.exists() and any(final_dir.iterdir()):
        logger.info("Reusing local hard-negative E5: %s", final_dir)
        return final_dir

    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader

    stage1_dir = paths.models/"legal_e5_finetuned"

    if not (cfg.resume and stage1_dir.exists() and any(stage1_dir.iterdir())):
        model = SentenceTransformer(cfg.e5_base_model, device=device)

        examples = []
        for _, r in train_df.iterrows():
            q = "query: " + r["question_clean"]
            p = "passage: " + r["canonical_citation"] + " | " + r["section_text_clean"]
            examples.append(InputExample(texts=[q, p]))

        loader = DataLoader(
            examples, shuffle=True, batch_size=cfg.train_batch_size
        )
        loss = losses.MultipleNegativesRankingLoss(model)

        logger.info("Training E5 stage 1...")
        model.fit(
            train_objectives=[(loader, loss)],
            epochs=cfg.stage1_epochs,
            warmup_steps=max(1, int(len(loader)*cfg.stage1_epochs*0.1)),
            optimizer_params={"lr": 2e-5},
            weight_decay=0.01,
            output_path=str(stage1_dir),
            show_progress_bar=True,
        )
        del model
        clear_memory()

    # Hard-negative mining
    model = SentenceTransformer(str(stage1_dir), device=device)

    corpus_texts = ["passage: " + x for x in corpus["retrieval_text"].tolist()]
    corpus_emb = model.encode(
        corpus_texts,
        batch_size=cfg.e5_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    query_texts = ["query: " + x for x in train_df["question_clean"].tolist()]
    query_emb = model.encode(
        query_texts,
        batch_size=cfg.e5_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    triples = []
    for i, row in tqdm(
        train_df.reset_index(drop=True).iterrows(),
        total=len(train_df),
        desc="Mining hard negatives",
        dynamic_ncols=True,
    ):
        scores = query_emb[i] @ corpus_emb.T
        top = np.argsort(scores)[::-1][:20]
        negative = None

        for idx in top:
            c = corpus.iloc[int(idx)]
            if c["canonical_citation"] != row["canonical_citation"]:
                negative = c["retrieval_text"]
                break

        if negative is not None:
            triples.append(
                InputExample(
                    texts=[
                        "query: " + row["question_clean"],
                        "passage: " + row["canonical_citation"] + " | " + row["section_text_clean"],
                        "passage: " + negative,
                    ]
                )
            )

    loader = DataLoader(
        triples, shuffle=True, batch_size=cfg.train_batch_size
    )
    loss = losses.TripletLoss(model=model)

    logger.info("Training E5 hard-negative stage...")
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=cfg.stage2_epochs,
        warmup_steps=max(1, int(len(loader)*cfg.stage2_epochs*0.1)),
        optimizer_params={"lr": 1e-5},
        weight_decay=0.01,
        output_path=str(final_dir),
        show_progress_bar=True,
    )

    del model, corpus_emb, query_emb
    clear_memory()
    return final_dir


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def bm25_retrieve(qa_df, corpus, top_k, logger):
    logger.info("BM25 retrieval...")
    bm25 = BM25Okapi([tokenize(x) for x in corpus["retrieval_text"]])
    out = []

    for q in tqdm(
        qa_df["question_clean"].tolist(),
        desc="BM25",
        dynamic_ncols=True,
    ):
        scores = bm25.get_scores(tokenize(q))
        out.append(np.argsort(scores)[::-1][:top_k].tolist())

    return out


def dense_retrieve(qa_df, corpus, model_path, cfg, device, logger):
    logger.info("Dense E5 retrieval...")
    model = SentenceTransformer(str(model_path), device=device)

    corpus_emb = model.encode(
        ["passage: " + x for x in corpus["retrieval_text"].tolist()],
        batch_size=cfg.e5_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    query_emb = model.encode(
        ["query: " + x for x in qa_df["question_clean"].tolist()],
        batch_size=cfg.e5_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    sims = query_emb @ corpus_emb.T
    rankings = [
        np.argsort(scores)[::-1][:cfg.dense_top_k].tolist()
        for scores in sims
    ]

    del model, query_emb
    clear_memory()
    return rankings, corpus_emb


def union_rankings(dense, sparse):
    merged_all = []

    for d, s in zip(dense, sparse):
        seen = set()
        merged = []
        for idx in list(d) + list(s):
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                merged.append(idx)
        merged_all.append(merged)

    return merged_all


def candidate_recall(qa_df, rankings, corpus):
    hits = []
    qa = qa_df.reset_index(drop=True)

    for i, r in qa.iterrows():
        citations = corpus.iloc[list(rankings[i])]["canonical_citation"].tolist()
        hits.append(int(r["canonical_citation"] in citations))

    return float(np.mean(hits)), hits


# ---------------------------------------------------------------------------
# M3 reranking with periodic resume checkpoint
# ---------------------------------------------------------------------------

def load_reranker(cfg, logger):
    from FlagEmbedding import FlagReranker
    logger.info("Loading %s", cfg.reranker_model)
    return FlagReranker(
        cfg.reranker_model,
        use_fp16=torch.cuda.is_available(),
    )


def rerank(
    qa_df, union_candidates, corpus, cfg, paths, logger, tag="validation"
):
    final_path = paths.checkpoints/f"m3_{tag}_rankings.pkl"
    partial_path = paths.checkpoints/f"m3_{tag}_rankings_partial.pkl"

    if cfg.resume and final_path.exists():
        saved = load_pickle(final_path)
        if len(saved) == len(qa_df):
            logger.info("Using completed reranking checkpoint.")
            return saved

    results = []
    start = 0

    if cfg.resume and partial_path.exists():
        results = load_pickle(partial_path)
        start = len(results)
        logger.info("Resuming reranking from %d/%d", start, len(qa_df))

    rr = load_reranker(cfg, logger)
    qa = qa_df.reset_index(drop=True)

    for i in tqdm(
        range(start, len(qa)),
        desc=f"BGE {tag}",
        dynamic_ncols=True,
    ):
        q = qa.iloc[i]["question_clean"]
        ids = [int(x) for x in union_candidates[i]]
        texts = corpus.iloc[ids]["retrieval_text"].tolist()
        pairs = [[q, p] for p in texts]

        with torch.inference_mode():
            try:
                scores = rr.compute_score(
                    pairs,
                    batch_size=cfg.reranker_batch_size,
                    max_length=cfg.reranker_max_length,
                    normalize=True,
                )
            except TypeError:
                scores = rr.compute_score(
                    pairs,
                    batch_size=cfg.reranker_batch_size,
                    normalize=True,
                )

        order = np.argsort(np.asarray(scores, dtype=float))[::-1]
        results.append([ids[int(j)] for j in order[:cfg.reranker_top_k]])

        # Save every 10 questions.
        if (i + 1) % 10 == 0:
            save_pickle(results, partial_path)

    save_pickle(results, final_path)
    save_pickle(results, partial_path)

    del rr
    clear_memory()
    return results


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(qa_df, rankings, corpus, ks=(1,3,5,10)):
    qa = qa_df.reset_index(drop=True)
    rows = []

    for i, r in qa.iterrows():
        gold = r["canonical_citation"]
        citations = corpus.iloc[list(rankings[i])]["canonical_citation"].tolist()

        rank = None
        for pos, c in enumerate(citations, 1):
            if c == gold:
                rank = pos
                break

        row = {
            "dataset_id": r.get("dataset_id", i),
            "language": r.get("language", "Unknown"),
            "question_type": r.get("question_type", "unknown"),
            "gold_citation": gold,
            "rank": rank,
            "reciprocal_rank": 1/rank if rank else 0.0,
        }

        for k in ks:
            row[f"Recall@{k}"] = int(rank is not None and rank <= k)

        rows.append(row)

    details = pd.DataFrame(rows)
    metrics = {f"Recall@{k}": float(details[f"Recall@{k}"].mean()) for k in ks}
    metrics["MRR@10"] = float(details["reciprocal_rank"].mean())
    return metrics, details


def failure_analysis(qa_df, union_candidates, final_rankings, corpus):
    qa = qa_df.reset_index(drop=True)
    rows = []

    for i, r in qa.iterrows():
        gold = r["canonical_citation"]

        pool = set(
            corpus.iloc[list(union_candidates[i])]["canonical_citation"].tolist()
        )
        top5 = set(
            corpus.iloc[list(final_rankings[i])[:5]]["canonical_citation"].tolist()
        )

        if gold in top5:
            label = "success"
        elif gold in pool:
            label = "ranking_failure"
        else:
            label = "candidate_retrieval_failure"

        rows.append({
            "dataset_id": r.get("dataset_id", i),
            "language": r.get("language", "Unknown"),
            "question_type": r.get("question_type", "unknown"),
            "failure_type": label,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report figures
# ---------------------------------------------------------------------------

def save_bar(labels, values, title, ylabel, path: Path):
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0,100)
    ax.grid(axis="y", alpha=.25)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x()+b.get_width()/2,
            v+1,
            f"{v:.2f}%",
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    candidate_metrics, metrics, details, candidate_df, failures, paths
):
    save_bar(
        ["BM25 Top-50", "E5 Top-50", "BM25+E5 Union"],
        [
            candidate_metrics["bm25"]*100,
            candidate_metrics["dense"]*100,
            candidate_metrics["union"]*100,
        ],
        "Validation Candidate Retrieval Performance",
        "Candidate Recall (%)",
        paths.figures/"candidate_retrieval.png",
    )

    ks = [1,3,5,10]
    vals = [metrics[f"Recall@{k}"]*100 for k in ks]

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(ks, vals, marker="o")
    ax.set_xticks(ks)
    ax.set_ylim(0,100)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (%)")
    ax.set_title("BGE v2-m3 Reranked Retrieval")
    ax.grid(alpha=.25)

    for x,y in zip(ks,vals):
        ax.text(x,y+1,f"{y:.2f}%",ha="center")

    fig.tight_layout()
    fig.savefig(paths.figures/"reranker_recall_at_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    save_bar(
        ["BM25\nTop-50", "E5\nTop-50", "Union\n~95", "BGE\nTop-5"],
        [
            candidate_metrics["bm25"]*100,
            candidate_metrics["dense"]*100,
            candidate_metrics["union"]*100,
            metrics["Recall@5"]*100,
        ],
        "Retrieval Performance Across Pipeline Stages",
        "Recall (%)",
        paths.figures/"pipeline_stages.png",
    )

    # Language grouped figure
    lang_union = candidate_df.groupby("language")["union_hit"].mean()*100
    lang_final = details.groupby("language")["Recall@5"].mean()*100
    groups = sorted(set(lang_union.index) | set(lang_final.index))

    x = np.arange(len(groups))
    w = .36
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x-w/2, [lang_union.get(g,np.nan) for g in groups], w, label="Candidate Union")
    ax.bar(x+w/2, [lang_final.get(g,np.nan) for g in groups], w, label="Reranked Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0,100)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Retrieval Performance by Language")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(paths.figures/"language_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Question type grouped figure
    qt_union = candidate_df.groupby("question_type")["union_hit"].mean()*100
    qt_final = details.groupby("question_type")["Recall@5"].mean()*100
    groups = sorted(set(qt_union.index) | set(qt_final.index))

    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x-w/2, [qt_union.get(g,np.nan) for g in groups], w, label="Candidate Union")
    ax.bar(x+w/2, [qt_final.get(g,np.nan) for g in groups], w, label="Reranked Top-5")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0,100)
    ax.set_ylabel("Recall (%)")
    ax.set_title("Retrieval Performance by Question Type")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(paths.figures/"question_type_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Failure distribution
    counts = failures["failure_type"].value_counts(normalize=True)*100
    labels = ["success", "ranking_failure", "candidate_retrieval_failure"]
    save_bar(
        ["Success", "Ranking Failure", "Candidate Failure"],
        [counts.get(x,0.0) for x in labels],
        "Retrieval Error Decomposition",
        "Validation Questions (%)",
        paths.figures/"failure_decomposition.png",
    )


# ---------------------------------------------------------------------------
# RAG evidence export
# ---------------------------------------------------------------------------

def export_rag(qa_df, rankings, corpus, out_path: Path, top_k=5):
    qa = qa_df.reset_index(drop=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in qa.iterrows():
            evidence = []
            for rank, idx in enumerate(list(rankings[i])[:top_k], 1):
                c = corpus.iloc[int(idx)]
                evidence.append({
                    "rank": rank,
                    "canonical_citation": c["canonical_citation"],
                    "section_text": c["section_text_clean"],
                })

            record = {
                "dataset_id": r.get("dataset_id", i),
                "question": r["question_clean"],
                "language": r.get("language", "Unknown"),
                "question_type": r.get("question_type", "unknown"),
                "gold_citation": r["canonical_citation"],
                "evidence": evidence,
            }
            f.write(json.dumps(record, ensure_ascii=False)+"\n")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def prepare(cfg, paths, logger):
    if cfg.split_dir:
        train_df, val_df, test_df = load_splits(
            Path(cfg.split_dir).expanduser().resolve()
        )
    elif cfg.resume and all(
        (paths.splits/x).exists()
        for x in ["train.jsonl","validation.jsonl","test.jsonl"]
    ):
        train_df, val_df, test_df = load_splits(paths.splits)
    else:
        raw = load_raw(dataset_path(cfg, paths, logger))
        clean = prepare_df(raw, logger)
        train_df, val_df, test_df = make_split(clean, cfg, logger)
        save_splits(train_df, val_df, test_df, paths.splits)

    corpus = build_corpus(train_df, val_df, test_df, logger)
    corpus.to_json(
        paths.data/"retrieval_corpus.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return train_df, val_df, test_df, corpus


def retrieve_validation(cfg, paths, val_df, corpus, model_path, device, logger):
    ckpt = paths.checkpoints/"validation_candidates.pkl"

    if cfg.resume and ckpt.exists():
        logger.info("Loading candidate checkpoint.")
        return load_pickle(ckpt)

    dense, corpus_emb = dense_retrieve(
        val_df, corpus, model_path, cfg, device, logger
    )
    np.save(paths.cache/"corpus_embeddings.npy", corpus_emb)
    del corpus_emb
    clear_memory()

    bm25 = bm25_retrieve(
        val_df, corpus, cfg.bm25_top_k, logger
    )
    union = union_rankings(dense, bm25)

    bm25_r, bm25_hits = candidate_recall(val_df, bm25, corpus)
    dense_r, dense_hits = candidate_recall(val_df, dense, corpus)
    union_r, union_hits = candidate_recall(val_df, union, corpus)

    out = {
        "bm25": bm25,
        "dense": dense,
        "union": union,
        "bm25_recall": bm25_r,
        "dense_recall": dense_r,
        "union_recall": union_r,
        "bm25_hits": bm25_hits,
        "dense_hits": dense_hits,
        "union_hits": union_hits,
    }
    save_pickle(out, ckpt)

    logger.info(
        "Candidate recall | BM25 %.2f%% | E5 %.2f%% | Union %.2f%% | Avg union %.2f",
        bm25_r*100,
        dense_r*100,
        union_r*100,
        np.mean([len(x) for x in union]),
    )
    return out


def save_validation_results(cfg, paths, val_df, corpus, cand, rankings, logger):
    metrics, details = evaluate(val_df, rankings, corpus)

    logger.info("BGE v2-m3 VALIDATION")
    for k,v in metrics.items():
        logger.info("%s = %.4f (%.2f%%)", k, v, v*100)

    lang = (
        details.groupby("language")
        .agg(N=("dataset_id","count"), Recall5=("Recall@5","mean"), MRR=("reciprocal_rank","mean"))
        .reset_index()
    )
    qtype = (
        details.groupby("question_type")
        .agg(N=("dataset_id","count"), Recall5=("Recall@5","mean"), MRR=("reciprocal_rank","mean"))
        .reset_index()
    )

    failures = failure_analysis(val_df, cand["union"], rankings, corpus)

    candidate_df = val_df[["dataset_id","language","question_type"]].reset_index(drop=True).copy()
    candidate_df["bm25_hit"] = cand["bm25_hits"]
    candidate_df["dense_hit"] = cand["dense_hits"]
    candidate_df["union_hit"] = cand["union_hits"]

    details.to_csv(paths.results/"m3_validation_details.csv", index=False)
    lang.to_csv(paths.results/"m3_by_language.csv", index=False)
    qtype.to_csv(paths.results/"m3_by_question_type.csv", index=False)
    failures.to_csv(paths.results/"failure_analysis.csv", index=False)
    candidate_df.to_csv(paths.results/"candidate_analysis.csv", index=False)

    candidate_metrics = {
        "bm25": cand["bm25_recall"],
        "dense": cand["dense_recall"],
        "union": cand["union_recall"],
    }
    save_json(
        {"candidate": candidate_metrics, "reranker": metrics},
        paths.results/"validation_metrics.json",
    )

    make_figures(
        candidate_metrics, metrics, details, candidate_df, failures, paths
    )

    export_rag(
        val_df, rankings, corpus,
        paths.results/"validation_rag_top5.jsonl",
        top_k=5,
    )

    full_ckpt = {
        "validation_df": val_df,
        "retrieval_corpus": corpus,
        "dense_validation_candidates": cand["dense"],
        "bm25_validation_candidates": cand["bm25"],
        "combined_validation_candidates": cand["union"],
        "m3_validation_rankings": rankings,
        "bm25_hits": cand["bm25_hits"],
        "dense_hits": cand["dense_hits"],
        "combined_hits": cand["union_hits"],
    }
    save_pickle(full_ckpt, paths.checkpoints/"retrieval_checkpoint.pkl")

    return metrics


def final_test_once(cfg, paths, test_df, corpus, model_path, device, logger):
    final_ckpt = paths.checkpoints/"FINAL_TEST_bundle.pkl"

    if cfg.resume and final_ckpt.exists():
        logger.warning("Final-test checkpoint already exists; reusing it.")
        return load_pickle(final_ckpt)

    logger.warning(
        "FINAL TEST STARTED. Do not use these results for further model tuning."
    )

    dense, _ = dense_retrieve(
        test_df, corpus, model_path, cfg, device, logger
    )
    bm25 = bm25_retrieve(
        test_df, corpus, cfg.bm25_top_k, logger
    )
    union = union_rankings(dense, bm25)
    ranked = rerank(
        test_df, union, corpus, cfg, paths, logger, tag="test"
    )

    metrics, details = evaluate(test_df, ranked, corpus)
    details.to_csv(paths.results/"FINAL_TEST_details.csv", index=False)
    save_json(metrics, paths.results/"FINAL_TEST_metrics.json")
    export_rag(
        test_df, ranked, corpus,
        paths.results/"FINAL_TEST_rag_top5.jsonl",
        top_k=5,
    )

    bundle = {
        "dense": dense,
        "bm25": bm25,
        "union": union,
        "rankings": ranked,
        "metrics": metrics,
    }
    save_pickle(bundle, final_ckpt)
    return bundle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage",
        choices=["prepare","train","retrieve","rerank","evaluate","all"],
        default="all",
    )
    p.add_argument("--work-dir", default="./artifacts")
    p.add_argument("--input-json")
    p.add_argument("--split-dir")
    p.add_argument("--hardneg-model")
    p.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--dense-top-k", type=int, default=50)
    p.add_argument("--bm25-top-k", type=int, default=50)
    p.add_argument("--reranker-top-k", type=int, default=10)
    p.add_argument("--e5-batch-size", type=int, default=32)
    p.add_argument("--reranker-batch-size", type=int, default=8)
    p.add_argument("--reranker-max-length", type=int, default=512)

    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--final-test", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        work_dir=args.work_dir,
        input_json=args.input_json,
        split_dir=args.split_dir,
        hardneg_model=args.hardneg_model,
        seed=args.seed,
        device=args.device,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        reranker_top_k=args.reranker_top_k,
        e5_batch_size=args.e5_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        resume=not args.no_resume,
        final_test=args.final_test,
    )

    paths = Paths.make(cfg.work_dir)
    logger = get_logger(paths)
    seed_all(cfg.seed)
    device = pick_device(cfg.device)

    logger.info("Device: %s", device)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    train_df, val_df, test_df, corpus = prepare(cfg, paths, logger)

    if args.stage == "prepare":
        return

    allow_train = args.stage in {"train","all"} or cfg.hardneg_model is not None
    if not allow_train:
        local_model = paths.models/"legal_e5_hardneg"
        if not local_model.exists():
            raise RuntimeError(
                "No hard-negative model found. Run --stage train or supply --hardneg-model."
            )

    model_path = train_e5_if_needed(
        cfg, paths, train_df, corpus, device, logger
    )

    if args.stage == "train":
        return

    cand = retrieve_validation(
        cfg, paths, val_df, corpus, model_path, device, logger
    )

    if args.stage == "retrieve":
        return

    rankings = rerank(
        val_df, cand["union"], corpus, cfg, paths, logger, tag="validation"
    )

    if args.stage == "rerank":
        return

    save_validation_results(
        cfg, paths, val_df, corpus, cand, rankings, logger
    )

    if cfg.final_test:
        final_test_once(
            cfg, paths, test_df, corpus, model_path, device, logger
        )

    logger.info("DONE")
    logger.info("Results: %s", paths.results)
    logger.info("Figures: %s", paths.figures)
    logger.info("Log: %s", paths.logs/"pipeline.log")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Existing checkpoints are preserved.")
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise
