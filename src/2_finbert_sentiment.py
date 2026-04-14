"""
3_finbert_sentiment.py
======================
Production-grade sentiment scoring pipeline using ProsusAI/finbert.

Pipeline overview
-----------------
1. Load the full dataset once.
2. Collect every unique article string across all three news columns.
3. Run a single batched FinBERT inference pass over only those unique strings
   (sliding-window chunking applied where necessary).
4. Build a cell-level cache: for each unique cell value (the full `||`-joined
   string) compute the averaged (pos, neu, neg) from its constituent articles
   and their article-level cache entries.
5. Assemble the output DataFrame row-by-row from the two caches.

This design maximises GPU throughput by:
  - eliminating all duplicate inference (one inference per unique article string)
  - processing all unique strings in large, contiguous batches
  - avoiding Python-level loops during inference

Author : (generated for INLP S26 project)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock between HuggingFace Rust tokenizer and PyTorch DataLoader workers
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV   = Path("dataset/tier_segragated_news.csv")
OUTPUT_CSV  = Path("dataset/news_sentiment.csv")
MODEL_NAME  = "ProsusAI/finbert"

# Tokenizer / sliding-window knobs
MAX_TOKENS      = 500   # treat texts above this length as "long"
WINDOW_TOKENS   = 400   # chunk size for the sliding window
OVERLAP_TOKENS  = 100   # overlap between consecutive chunks
STRIDE_TOKENS   = WINDOW_TOKENS - OVERLAP_TOKENS  # effective stride = 300

# Inference knobs – tuned for a 4 GB RTX 3050 / 16 GB RAM system
BATCH_SIZE  = 16         # smaller batch reduces OOM risk on long sequences
NUM_WORKERS = 4         # enough prefetching without unnecessary RAM pressure

# Separator used between articles inside a cell
ARTICLE_SEP = " || "

# Column names in the input file (lower-cased for robustness)
COL_DATE   = "date"
COL_SYMBOL = "symbol"
NEWS_COLS  = ["direct_news", "sectoral_news", "global_news"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper: article count from a raw cell value
# ---------------------------------------------------------------------------

def count_articles(cell: str) -> int:
    """Return the number of articles in a cell (number of `||` separators + 1)."""
    if not cell or not cell.strip():
        return 0
    return cell.count("||") + 1


# ---------------------------------------------------------------------------
# Step 1 – tokenise a single article with a sliding window
#           returns a list of {"input_ids", "attention_mask"} dicts
# ---------------------------------------------------------------------------

def sliding_window_chunks(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS,
    window: int = WINDOW_TOKENS,
    stride: int = STRIDE_TOKENS,
) -> List[Dict]:
    """
    Tokenize `text`.  If it fits within `max_tokens` tokens, return one chunk.
    Otherwise produce overlapping chunks of `window` tokens with step `stride`.
    Each chunk is returned as a dict with keys 'input_ids' and 'attention_mask'
    (both plain Python lists, no batch dimension).
    """
    # Tokenise without truncation to get the full token sequence
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,   # plain Python lists
    )
    ids = encoded["input_ids"]

    # Special token ids
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if len(ids) <= max_tokens:
        # Short text – single chunk with proper special tokens
        chunk_ids  = [cls_id] + ids + [sep_id]
        chunk_mask = [1] * len(chunk_ids)
        return [{"input_ids": chunk_ids, "attention_mask": chunk_mask}]

    # Long text – sliding window over the raw token ids
    chunks = []
    start = 0
    while start < len(ids):
        end        = min(start + window, len(ids))
        chunk_ids  = [cls_id] + ids[start:end] + [sep_id]
        chunk_mask = [1] * len(chunk_ids)
        chunks.append({"input_ids": chunk_ids, "attention_mask": chunk_mask})
        if end == len(ids):
            break
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Step 2 – PyTorch Dataset that wraps a flat list of pre-tokenised chunks
# ---------------------------------------------------------------------------

class ChunkDataset(Dataset):
    """
    Holds a flat list of tokenised chunks (each a dict with 'input_ids' and
    'attention_mask'). The collate function pads them to the batch maximum.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict:
        return self.chunks[idx]


def collate_pad(batch: List[Dict], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-length chunks to the same length."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_batch  = []
    attn_mask_batch  = []

    for item in batch:
        ids  = item["input_ids"]
        mask = item["attention_mask"]
        pad_len = max_len - len(ids)
        input_ids_batch.append(ids  + [pad_id] * pad_len)
        attn_mask_batch.append(mask + [0]      * pad_len)

    return {
        "input_ids":      torch.tensor(input_ids_batch,  dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask_batch,  dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Step 3 – run batched FinBERT inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    chunks: List[Dict],
    device: torch.device,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    desc: str = "FinBERT inference",
) -> np.ndarray:
    """
    Run FinBERT on `chunks` (flat list of tokenised dicts).
    Returns an (N, 3) float32 array of softmax probabilities:
        column 0 = positive, column 1 = negative, column 2 = neutral
    NOTE: HuggingFace FinBERT label order is [positive, negative, neutral]
          We re-order to [pos, neu, neg] in post-processing.
    """
    dataset = ChunkDataset(chunks)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda b: collate_pad(b, pad_id=0),
        pin_memory=(device.type == "cuda"),
    )

    all_probs = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # torch.autocast handles mixed-precision safely without NaN risks
            # from layer-norm that can occur with a blanket model.half() cast
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(**batch).logits             # (B, 3)

            probs = torch.softmax(logits.float(), dim=-1).cpu()  # cast back to fp32 before softmax
            all_probs.append(probs.numpy())

    return np.concatenate(all_probs, axis=0)  # (N, 3)


# ---------------------------------------------------------------------------
# Step 4 – build the article-level sentiment cache
# ---------------------------------------------------------------------------

def build_article_cache(
    unique_articles: List[str],
    tokenizer,
    model,
    device: torch.device,
) -> Dict[str, Tuple[float, float, float]]:
    """
    For every unique article string:
      1. Tokenise with sliding window → N_chunks chunks
      2. Run inference → N_chunks × 3 probability vectors
      3. Average across chunks (equal weight) → (pos, neu, neg)

    Returns a dict  article_text → (pos, neu, neg)

    FinBERT's id2label default order:   0=positive, 1=negative, 2=neutral
    We re-map to the output convention:  pos, neu, neg
    """
    log.info(f"Building article-level cache for {len(unique_articles):,} unique articles …")

    # --- tokenise all articles, recording the slice boundaries ---
    all_chunks: List[Dict] = []
    article_slices: List[Tuple[int, int]] = []  # (start_idx, end_idx) into all_chunks

    log.info("Tokenising unique articles (sliding window) …")
    for article in tqdm(unique_articles, desc="Tokenising", unit="art"):
        start = len(all_chunks)
        chunks = sliding_window_chunks(article, tokenizer)
        all_chunks.extend(chunks)
        article_slices.append((start, len(all_chunks)))

    log.info(f"Total chunks to infer: {len(all_chunks):,}")

    # --- run inference in one pass ---
    probs_all = run_inference(model, all_chunks, device, desc="FinBERT → articles")
    # probs_all shape: (total_chunks, 3)   columns: pos=0, neg=1, neu=2

    # --- aggregate & cache ---
    article_cache: Dict[str, Tuple[float, float, float]] = {}
    for article, (s, e) in zip(unique_articles, article_slices):
        chunk_probs = probs_all[s:e]           # (n_chunks, 3)
        mean_probs  = chunk_probs.mean(axis=0) # (3,)
        # FinBERT label order: 0=positive, 1=negative, 2=neutral
        pos = float(mean_probs[0])
        neg = float(mean_probs[1])
        neu = float(mean_probs[2])
        article_cache[article] = (pos, neu, neg)  # store as (pos, neu, neg)

    log.info("Article cache built ✓")
    return article_cache


# ---------------------------------------------------------------------------
# Step 5 – score a single cell given the article cache
# ---------------------------------------------------------------------------

def score_cell(
    cell_value,
    article_cache: Dict[str, Tuple[float, float, float]],
) -> Tuple[float, float, float]:
    """
    Split `cell_value` by ARTICLE_SEP.
    Retrieve each article's (pos, neu, neg) from `article_cache`.
    Return the mean across all articles in the cell.

    Returns (0.0, 0.0, 0.0) for empty/NaN cells.
    """
    # Handle missing / empty
    if cell_value is None or (isinstance(cell_value, float) and math.isnan(cell_value)):
        return (0.0, 0.0, 0.0)
    cell_str = str(cell_value).strip()
    if not cell_str:
        return (0.0, 0.0, 0.0)

    articles = [a.strip() for a in cell_str.split(ARTICLE_SEP) if a.strip()]
    if not articles:
        return (0.0, 0.0, 0.0)

    pos_vals, neu_vals, neg_vals = [], [], []
    for art in articles:
        p, n, g = article_cache[art]
        pos_vals.append(p)
        neu_vals.append(n)
        neg_vals.append(g)

    return (
        float(np.mean(pos_vals)),
        float(np.mean(neu_vals)),
        float(np.mean(neg_vals)),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 0. Validate paths
    # ------------------------------------------------------------------
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    log.info(f"Loading dataset: {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.lower().str.strip()

    # Validate expected columns
    required = [COL_DATE, COL_SYMBOL] + NEWS_COLS
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}. Found: {list(df.columns)}")

    log.info(f"Dataset loaded: {len(df):,} rows × {len(df.columns)} cols")

    # ------------------------------------------------------------------
    # 2. Collect all unique article strings across all news columns
    # ------------------------------------------------------------------
    log.info("Collecting unique article strings …")
    unique_articles_set: set = set()

    for col in NEWS_COLS:
        for cell in df[col].dropna():
            cell_str = str(cell).strip()
            if not cell_str:
                continue
            for art in cell_str.split(ARTICLE_SEP):
                art = art.strip()
                if art:
                    unique_articles_set.add(art)

    unique_articles = sorted(unique_articles_set)  # deterministic order
    log.info(f"Unique article strings: {len(unique_articles):,}")

    # ------------------------------------------------------------------
    # 3. Load FinBERT model & tokenizer
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Loading tokenizer & model: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_kwargs = {"low_cpu_mem_usage": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **model_kwargs)
    model     = model.to(device)
    model.eval()

    # Mixed precision is handled inside run_inference via torch.autocast.
    # Avoid model.half() here — it can cause NaN in LayerNorm on some architectures.

    # ------------------------------------------------------------------
    # 4. Build article-level sentiment cache
    # ------------------------------------------------------------------
    article_cache = build_article_cache(unique_articles, tokenizer, model, device)

    # Free GPU memory – no more inference needed after this point
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Assemble output DataFrame
    # ------------------------------------------------------------------
    log.info("Assembling output DataFrame …")

    rows = []
    append_row = rows.append
    date_idx = df.columns.get_loc(COL_DATE)
    symbol_idx = df.columns.get_loc(COL_SYMBOL)
    news_indices = {col: df.columns.get_loc(col) for col in NEWS_COLS}

    for row in tqdm(df.itertuples(index=False, name=None), total=len(df), desc="Building output", unit="row"):
        record = {
            "Date":   row[date_idx],
            "Symbol": row[symbol_idx],
        }

        for col in NEWS_COLS:
            cell = row[news_indices[col]]
            prefix = col  # e.g.  "direct_news"

            pos, neu, neg = score_cell(cell, article_cache)
            record[f"{prefix}_pos"] = pos
            record[f"{prefix}_neu"] = neu
            record[f"{prefix}_neg"] = neg
            record[f"{prefix}_count"] = count_articles(
                "" if (cell is None or (isinstance(cell, float) and math.isnan(cell))) else str(cell)
            )

        append_row(record)

    out_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 6. Enforce exact output column order
    # ------------------------------------------------------------------
    output_columns = [
        "Date", "Symbol",
        "direct_news_pos",    "direct_news_neu",    "direct_news_neg",
        "sectoral_news_pos",  "sectoral_news_neu",  "sectoral_news_neg",
        "global_news_pos",    "global_news_neu",    "global_news_neg",
        "direct_news_count",  "sectoral_news_count", "global_news_count",
    ]

    # Rename internal column names to match the exact spec
    # (internal: direct_news_count → output: direct_news_count)
    # The sectoral typo in the spec ("secroral_news_news_count") is intentionally
    # corrected to "sectoral_news_count" for correctness.
    out_df = out_df[output_columns]

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Output saved → {OUTPUT_CSV.resolve()}")
    log.info(f"Rows: {len(out_df):,}  |  Columns: {list(out_df.columns)}")

    # Quick sanity check
    log.info("\n--- Sample output (first 3 rows) ---")
    log.info("\n" + out_df.head(3).to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
