"""
Qwen Local LLM Financial News Segregation Pipeline
==================================================
Feature parity with the OpenAI pipeline for:
- Resume via checkpoint
- Failed row retry lifecycle
- Mode menu (NER-only / CSV-only / Full)
- CSV aggregation with inline sector normalization

Qwen/Ollama remains the extraction backend.
"""

import os
import random
import re
import gc
import json
import signal
import threading
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import ollama

warnings.filterwarnings("ignore")

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_WORKERS = 10
CHUNK_SIZE = 1000
MAX_RETRIES = 3
OLLAMA_MODEL = "qwen2.5:3b"
TEMPERATURE = 0.0

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "dataset" / "processed_news_dataset.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "dataset" / "news_segregation_checkpoints" / "raw_responses.jsonl"
FAILED_ROWS_FILE = PROJECT_ROOT / "dataset" / "news_segregation_checkpoints" / "failed_rows.txt"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "tier_segregated_news.csv"
MAPPING_CACHE_FILE = PROJECT_ROOT / "dataset" / "news_segregation_checkpoints" / "sector_mapping_cache.json"

SECTOR_GPT_BATCH_SIZE = 100
SECTOR_GPT_MODEL = "gpt-4o-mini"
OTHERS_SECTOR = "Others"


# ============================================================================
# NIFTY 50 COMPANY DATA (ALIGNED WITH 1a)
# ============================================================================

NIFTY_50_DATA = {
    # IT Sector
    "TCS": {"aliases": ["tcs", "tata consultancy", "tata consultancy services"], "sector": "IT"},
    "INFY": {"aliases": ["infosys", "infy"], "sector": "IT"},
    "HCLTECH": {"aliases": ["hcl tech", "hcl technologies", "hcltech"], "sector": "IT"},
    "WIPRO": {"aliases": ["wipro"], "sector": "IT"},
    "TECHM": {"aliases": ["tech mahindra", "techm"], "sector": "IT"},
    "LTIM": {"aliases": ["ltimindtree", "lti mindtree", "ltim", "l&t infotech"], "sector": "IT"},

    # Banking Sector
    "HDFCBANK": {"aliases": ["hdfc bank", "hdfcbank"], "sector": "Banking"},
    "ICICIBANK": {"aliases": ["icici bank", "icicibank"], "sector": "Banking"},
    "SBIN": {"aliases": ["sbi", "state bank of india", "state bank"], "sector": "Banking"},
    "AXISBANK": {"aliases": ["axis bank", "axisbank"], "sector": "Banking"},
    "KOTAKBANK": {"aliases": ["kotak mahindra bank", "kotak bank", "kotakbank"], "sector": "Banking"},
    "INDUSINDBK": {"aliases": ["indusind bank", "indusindbk"], "sector": "Banking"},
    "BANKBARODA": {"aliases": ["bank of baroda", "bob", "bankbaroda"], "sector": "Banking"},
    "PNB": {"aliases": ["punjab national bank", "pnb"], "sector": "Banking"},
    "CANBK": {"aliases": ["canara bank", "canbk"], "sector": "Banking"},

    # Financial Services
    "BAJFINANCE": {"aliases": ["bajaj finance", "bajfinance"], "sector": "Financial Services"},
    "BAJAJFINSV": {"aliases": ["bajaj finserv", "bajajfinsv"], "sector": "Financial Services"},
    "HDFCLIFE": {"aliases": ["hdfc life", "hdfclife"], "sector": "Financial Services"},
    "SBILIFE": {"aliases": ["sbi life", "sbilife"], "sector": "Financial Services"},

    # Auto Sector
    "MARUTI": {"aliases": ["maruti suzuki", "maruti", "maruti suzuki india"], "sector": "Auto"},
    "M&M": {"aliases": ["mahindra & mahindra", "mahindra", "m&m", "m and m"], "sector": "Auto"},
    "TATAMOTORS": {"aliases": ["tata motors", "tatamotors"], "sector": "Auto"},
    "BAJAJ-AUTO": {"aliases": ["bajaj auto", "bajaj-auto"], "sector": "Auto"},
    "EICHERMOT": {"aliases": ["eicher motors", "eicher", "royal enfield"], "sector": "Auto"},
    "HEROMOTOCO": {"aliases": ["hero motocorp", "hero", "heromotoco"], "sector": "Auto"},

    # FMCG Sector
    "HINDUNILVR": {"aliases": ["hindustan unilever", "hul", "hindunilvr"], "sector": "FMCG"},
    "ITC": {"aliases": ["itc", "itc limited"], "sector": "FMCG"},
    "NESTLEIND": {"aliases": ["nestle india", "nestle", "nestleind"], "sector": "FMCG"},
    "BRITANNIA": {"aliases": ["britannia", "britannia industries"], "sector": "FMCG"},
    "DABUR": {"aliases": ["dabur", "dabur india"], "sector": "FMCG"},
    "TATACONSUM": {"aliases": ["tata consumer", "tata consumer products", "tataconsum"], "sector": "FMCG"},

    # Pharma Sector
    "SUNPHARMA": {"aliases": ["sun pharma", "sun pharmaceutical", "sunpharma"], "sector": "Pharma"},
    "DRREDDY": {"aliases": ["dr reddy", "dr reddy's", "drreddys", "drreddy"], "sector": "Pharma"},
    "CIPLA": {"aliases": ["cipla"], "sector": "Pharma"},
    "DIVISLAB": {"aliases": ["divi's lab", "divis lab", "divislab"], "sector": "Pharma"},

    # Energy & Power
    "RELIANCE": {"aliases": ["reliance", "reliance industries", "ril"], "sector": "Energy"},
    "ONGC": {"aliases": ["ongc", "oil and natural gas"], "sector": "Energy"},
    "BPCL": {"aliases": ["bpcl", "bharat petroleum"], "sector": "Energy"},
    "IOC": {"aliases": ["ioc", "indian oil", "indian oil corp"], "sector": "Energy"},
    "NTPC": {"aliases": ["ntpc"], "sector": "Power"},
    "POWERGRID": {"aliases": ["power grid", "powergrid"], "sector": "Power"},

    # Metals & Mining
    "TATASTEEL": {"aliases": ["tata steel", "tatasteel"], "sector": "Metals"},
    "HINDALCO": {"aliases": ["hindalco", "hindalco industries"], "sector": "Metals"},
    "JSWSTEEL": {"aliases": ["jsw steel", "jswsteel"], "sector": "Metals"},
    "COALINDIA": {"aliases": ["coal india", "coalindia"], "sector": "Metals"},

    # Telecom
    "BHARTIARTL": {"aliases": ["bharti airtel", "airtel", "bhartiartl"], "sector": "Telecom"},

    # Cement
    "ULTRACEMCO": {"aliases": ["ultratech cement", "ultratech", "ultracemco"], "sector": "Cement"},
    "GRASIM": {"aliases": ["grasim", "grasim industries"], "sector": "Cement"},

    # Infrastructure
    "LT": {"aliases": ["l&t", "larsen & toubro", "larsen and toubro", "lt"], "sector": "Infrastructure"},
    "ADANIPORTS": {"aliases": ["adani ports", "adaniports", "apsez"], "sector": "Infrastructure"},
    "ADANIENT": {
        "aliases": ["adani enterprises", "adani enterprises ltd", "adani enterprises ltd.", "adani ent", "ael"],
        "sector": "Infrastructure",
    },

    # Others
    "ASIANPAINT": {"aliases": ["asian paints", "asianpaint"], "sector": "Paints"},
    "TITAN": {"aliases": ["titan", "titan company"], "sector": "Retail"},
    "APOLLOHOSP": {"aliases": ["apollo hospital", "apollo hospitals", "apollohosp"], "sector": "Healthcare"},
    "INDIGO": {"aliases": ["indigo", "interglobe aviation"], "sector": "Aviation"},
}

SECTOR_KEYWORDS = {
    "IT": ["it sector", "software", "technology", "digital", "cloud computing", "ai", "automation", "deal win", "tech spend"],
    "Banking": ["banking", "banks", "npa", "lending", "deposits", "credit", "casa", "psu bank", "bad loans", "nbfc"],
    "Financial Services": ["insurance", "mutual fund", "financial services", "amc", "life insurance"],
    "Auto": ["automobile", "automotive", "vehicle", "cars", "ev", "electric vehicle", "two-wheeler", "commercial vehicle", "tractor", "sales data", "auto sales"],
    "FMCG": ["fmcg", "consumer goods", "rural demand", "staples", "rural volume", "consumption"],
    "Pharma": ["pharmaceutical", "pharma", "drug", "medicine", "healthcare", "fda", "usfda", "api", "generics", "clinical trials"],
    "Energy": ["oil", "gas", "petroleum", "crude", "refinery", "brent", "o&m"],
    "Power": ["power", "electricity", "renewable energy", "solar", "wind energy", "discom", "power generation"],
    "Metals": ["steel", "aluminium", "aluminum", "copper", "metals", "mining", "iron ore", "base metals"],
    "Telecom": ["telecom", "telecommunication", "5g", "spectrum", "arpu", "telecom tariffs", "broadband"],
    "Cement": ["cement", "construction material", "clinker", "capacity expansion"],
    "Infrastructure": ["infrastructure", "construction", "ports", "highways", "capex", "nhai", "toll"],
    "Aviation": ["aviation", "airlines", "passenger traffic", "cargo traffic", "load factor"],
}

GLOBAL_KEYWORDS = [
    "rbi", "reserve bank", "repo rate", "monetary policy", "interest rate", "fed", "federal reserve", "fomc", "ecb", "bond yield",
    "budget", "fiscal policy", "gst", "tax", "inflation", "cpi", "wpi", "gdp", "economic growth", "recession", "pmi", "iip", "manufacturing pmi",
    "us market", "dow jones", "nasdaq", "s&p 500", "wall street", "nifty", "sensex", "nse", "bse", "sebi", "stock market", "foreign investment", "fii", "fpi", "dii", "fdi", "market crash", "market rally",
    "crude oil", "brent crude", "gold price", "dollar", "rupee", "inr", "usd", "opec",
    "covid", "pandemic", "lockdown", "ukraine", "russia", "china", "geopolitical", "trade war", "tariff", "sanctions", "war", "monsoon", "elections",
]

VALID_SECTORS: Set[str] = set(SECTOR_KEYWORDS.keys()) | {OTHERS_SECTOR}
STANDARD_SECTOR_LOWER: Dict[str, str] = {s.lower(): s for s in VALID_SECTORS}


SYSTEM_PROMPT = """You are a financial news analyzer. Extract information into exactly 3 categories and return valid JSON.

Categories:
1. "direct": Companies explicitly mentioned with their impact
2. "sectoral": Industry sectors or supply chains affected
3. "global": Macro/geopolitical events affecting markets

Rules:
- Return ONLY valid JSON with these 3 keys
- If a category has no information, return empty array []
- Be concise in summaries (2 to 3 sentences max)
- Extract ALL relevant information

Example 1:
Input: "TCS reported strong Q3 growth with 15% YoY increase. Tech sector benefits from digital transformation."
Output:
{
  "direct": [{"company": "TCS", "summary": "TCS reports strong Q3 growth with 15% YoY increase"}],
  "sectoral": [{"sector": "IT", "summary": "Tech sector benefits from digital transformation initiatives"}],
  "global": []
}

Example 2:
Input: "RBI increases repo rate by 25 bps to combat inflation. This will impact banking sector lending."
Output:
{
  "direct": [],
  "sectoral": [{"sector": "Banking", "summary": "RBI increases repo rate by 25 bps, impacting lending rates"}],
  "global": [{"event": "Monetary Policy", "summary": "RBI hikes repo rate to combat inflation"}]
}

Example 3:
Input: "Crude oil prices surge 5% amid geopolitical tensions. Energy stocks rally."
Output:
{
  "direct": [],
  "sectoral": [{"sector": "Energy", "summary": "Energy stocks rally on crude price surge"}],
  "global": [{"event": "Crude Oil", "summary": "Crude oil prices surge 5% amid geopolitical tensions"}]
}

Example 4:
Input: "Reliance Industries announces expansion in retail business. Maruti Suzuki launches new EV model."
Output:
{
  "direct": [
    {"company": "Reliance", "summary": "Reliance Industries announces expansion in retail business"},
    {"company": "Maruti", "summary": "Maruti Suzuki launches new EV model"}
  ],
  "sectoral": [{"sector": "Auto", "summary": "Auto sector embraces EV transition"}],
  "global": []
}

Rule you have to follow:
1. Rule against inventing companies: "NEVER assign a company to the 'direct' category unless that specific company is explicitly named in the text. Do not guess."
2. Rule against copying examples: "The examples provided above are for formatting only. Do NOT include them in your output."
3. Rule for sectors: "For the 'sectoral' category, you MUST use ONLY these exact sector names: IT, Banking, Financial Services, Auto, FMCG, Pharma, Energy, Power, Metals, Telecom, Cement, Infrastructure, Aviation."

Now analyze the following news article and return only the JSON output:"""


class QwenFinancialNERPipeline:
    """
    Local LLM pipeline for financial news NER using Ollama + Qwen.
    Produces checkpoint output compatible with the OpenAI pipeline.
    """

    def __init__(self):
        self._checkpoint_lock = threading.Lock()
        self._failed_lock = threading.Lock()
        self.shutdown_event = threading.Event()

        self.company_patterns = self._build_company_patterns()
        self.sector_patterns = self._build_sector_patterns()
        self.global_patterns = self._build_global_patterns()

        self.company_to_ticker = self._build_company_mapping()
        self.sector_to_tickers = self._build_sector_mapping()
        self.all_tickers = list(NIFTY_50_DATA.keys())

        self.adani_enterprises_pattern = re.compile(
            r"\b(?:adani enterprises(?:\s+ltd\.?)?|adani ent|ael)\b",
            re.IGNORECASE,
        )
        self.adani_ports_pattern = re.compile(
            r"\b(?:adani ports(?:\s*(?:&|and)\s*sez|\s+sez)?|apsez)\b",
            re.IGNORECASE,
        )

        self.processed_indices: Set[int] = set()

        api_key = os.environ.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        if api_key and OpenAI is not None:
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception:
                self.openai_client = None

    def _build_company_patterns(self) -> List[re.Pattern]:
        patterns: List[re.Pattern] = []
        for data in NIFTY_50_DATA.values():
            for alias in data["aliases"]:
                patterns.append(re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE))
        return patterns

    def _build_sector_patterns(self) -> List[re.Pattern]:
        patterns: List[re.Pattern] = []
        for keywords in SECTOR_KEYWORDS.values():
            for keyword in keywords:
                patterns.append(re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE))
        return patterns

    def _build_global_patterns(self) -> List[re.Pattern]:
        return [re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE) for keyword in GLOBAL_KEYWORDS]

    def _build_company_mapping(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for ticker, data in NIFTY_50_DATA.items():
            for alias in data["aliases"]:
                mapping[alias.lower()] = ticker
        return mapping

    def _build_sector_mapping(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = defaultdict(list)
        for ticker, data in NIFTY_50_DATA.items():
            mapping[data["sector"]].append(ticker)
        return dict(mapping)

    @staticmethod
    def sanitize_text(text: str) -> str:
        if not text:
            return ""
        text = text.encode("utf-8", "ignore").decode("utf-8")
        text = re.sub(r"[^\x20-\x7E\n\t]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def is_relevant(self, text: str) -> bool:
        if not text or pd.isna(text):
            return False
        for pattern in self.company_patterns:
            if pattern.search(text):
                return True
        for pattern in self.sector_patterns:
            if pattern.search(text):
                return True
        for pattern in self.global_patterns:
            if pattern.search(text):
                return True
        return False

    def extract_entities(self, text: str) -> Optional[Dict]:
        text = self.sanitize_text(text)
        if not text:
            return {"direct": [], "sectoral": [], "global": []}

        for attempt in range(MAX_RETRIES):
            try:
                response = ollama.chat(
                    model=OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": text},
                    ],
                    format="json",
                    options={"temperature": TEMPERATURE},
                )

                content = (response.get("message", {}).get("content") or "").strip()
                content = re.sub(r"^```json\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
                result = json.loads(content)

                if all(key in result for key in ["direct", "sectoral", "global"]):
                    return result
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                    continue
                return None

        return None

    def normalize_extraction(self, extraction: Dict) -> Dict:
        normalized = {
            "direct": defaultdict(list),
            "sectoral": defaultdict(list),
            "global": defaultdict(list),
        }

        for item in extraction.get("direct", []):
            if not isinstance(item, dict):
                continue
            company = (item.get("company") or "").lower()
            summary = item.get("summary") or ""

            ticker = None
            for alias, tick in self.company_to_ticker.items():
                if alias in company or company in alias:
                    ticker = tick
                    break

            if ticker and summary:
                normalized["direct"][ticker].append(summary)

        for item in extraction.get("sectoral", []):
            if not isinstance(item, dict):
                continue
            sector = item.get("sector") or ""
            summary = item.get("summary") or ""
            if sector and summary:
                normalized["sectoral"][sector].append(summary)

        for item in extraction.get("global", []):
            if not isinstance(item, dict):
                continue
            event = (item.get("event") or item.get("summary") or "")[:50]
            summary = item.get("summary") or ""
            if summary:
                normalized["global"][event].append(summary)

        return {
            "direct": dict(normalized["direct"]),
            "sectoral": dict(normalized["sectoral"]),
            "global": dict(normalized["global"]),
        }

    def write_checkpoint(self, row_index: int, date: str, extracted_news: Dict, skipped: bool = False) -> None:
        with self._checkpoint_lock:
            with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                checkpoint_data = {
                    "row_index": row_index,
                    "date": date,
                    "extracted_news": extracted_news,
                }
                if skipped:
                    checkpoint_data["skipped"] = True
                f.write(json.dumps(checkpoint_data) + "\n")

    def write_failed_row(self, row_index: int, reason: str = "unknown") -> None:
        with self._failed_lock:
            with open(FAILED_ROWS_FILE, "a", encoding="utf-8") as f:
                f.write(f"{row_index}|{reason}\n")

    def load_processed_indices(self) -> Set[int]:
        processed: Set[int] = set()
        skipped_count = 0
        if CHECKPOINT_FILE.exists():
            print(f"[Resume] Loading checkpoint from {CHECKPOINT_FILE}...")
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed.add(int(data["row_index"]))
                        if data.get("skipped"):
                            skipped_count += 1
                    except Exception:
                        continue
            print(f"[Resume] Found {len(processed):,} already processed articles ({skipped_count:,} skipped as irrelevant)")
        return processed

    def load_failed_indices(self) -> Set[int]:
        failed: Set[int] = set()
        if FAILED_ROWS_FILE.exists():
            with open(FAILED_ROWS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        failed.add(int(line.split("|")[0]))
                    except Exception:
                        continue
        return failed

    def remove_from_failed_file(self, row_index: int) -> None:
        if not FAILED_ROWS_FILE.exists():
            return
        with self._failed_lock:
            with open(FAILED_ROWS_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(FAILED_ROWS_FILE, "w", encoding="utf-8") as f:
                for line in lines:
                    try:
                        idx = int(line.strip().split("|")[0])
                        if idx != row_index:
                            f.write(line)
                    except Exception:
                        f.write(line)

    def process_article(self, row_index: int, row: pd.Series) -> bool:
        try:
            if self.shutdown_event.is_set():
                return False
            if row_index in self.processed_indices:
                return True

            date = str(row.get("date", ""))
            text = f"{row.get('title', '')} {row.get('news', '')}"

            if not self.is_relevant(text):
                empty_news = {"direct": {}, "sectoral": {}, "global": {}}
                self.write_checkpoint(row_index, date, empty_news, skipped=True)
                return True

            extraction = self.extract_entities(text)
            if extraction is None:
                self.write_failed_row(row_index, reason="llm_failure")
                return False

            extracted_news = self.normalize_extraction(extraction)
            self.write_checkpoint(row_index, date, extracted_news)
            return True
        except Exception as e:
            print(f"Error processing article {row_index}: {e}")
            self.write_failed_row(row_index, reason=f"exception:{type(e).__name__}")
            return False

    def process_chunk(self, chunk: pd.DataFrame, pbar) -> int:
        success_count = 0
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for row_index, row in chunk.iterrows():
                if self.shutdown_event.is_set():
                    break
                future = executor.submit(self.process_article, row_index, row)
                futures[future] = row_index

            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                except Exception as e:
                    print(f"Thread error for row {futures[future]}: {e}")
                finally:
                    pbar.update(1)
                if self.shutdown_event.is_set():
                    continue
        return success_count

    def process_failed_article(self, row_index: int, row: pd.Series) -> bool:
        success = self.process_article(row_index, row)
        if success:
            self.remove_from_failed_file(row_index)
        return success

    def process_failed_chunk(self, chunk: pd.DataFrame, pbar) -> int:
        success_count = 0
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for row_index, row in chunk.iterrows():
                if self.shutdown_event.is_set():
                    break
                future = executor.submit(self.process_failed_article, row_index, row)
                futures[future] = row_index

            for future in as_completed(futures):
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                except Exception as e:
                    print(f"Retry thread error for row {futures[future]}: {e}")
                finally:
                    pbar.update(1)
        return success_count

    def keyword_match_sector(self, label: str) -> Optional[str]:
        label_lower = (label or "").lower()
        matches: Dict[str, int] = {}
        for sector, keywords in SECTOR_KEYWORDS.items():
            for kw in keywords:
                if kw in label_lower:
                    if sector not in matches or len(kw) > matches[sector]:
                        matches[sector] = len(kw)
        if not matches:
            return None
        return max(matches, key=lambda s: matches[s])

    def canonical_sector(self, label: str) -> Optional[str]:
        if label in VALID_SECTORS:
            return label
        ci = STANDARD_SECTOR_LOWER.get((label or "").lower())
        if ci:
            return ci
        return self.keyword_match_sector(label)

    def load_mapping_cache(self) -> Dict[str, str]:
        if MAPPING_CACHE_FILE.exists():
            try:
                with open(MAPPING_CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if isinstance(cache, dict):
                    clean_cache: Dict[str, str] = {}
                    for k, v in cache.items():
                        if not isinstance(k, str) or not isinstance(v, str):
                            continue
                        mapped = v if v in VALID_SECTORS else STANDARD_SECTOR_LOWER.get(v.lower(), OTHERS_SECTOR)
                        clean_cache[k] = mapped
                    return clean_cache
            except Exception:
                pass
        return {}

    def save_mapping_cache(self, cache: Dict[str, str]) -> None:
        MAPPING_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MAPPING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def fix_cache_with_keywords(self, cache: Dict[str, str]) -> int:
        corrected = 0
        for label in list(cache.keys()):
            if cache[label] == OTHERS_SECTOR:
                match = self.keyword_match_sector(label)
                if match:
                    cache[label] = match
                    corrected += 1
        return corrected

    def collect_unknown_sectors(self, cache: Dict[str, str]) -> Set[str]:
        unknown: Set[str] = set()
        if not CHECKPOINT_FILE.exists():
            return unknown

        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sectoral = data.get("extracted_news", {}).get("sectoral", {})
                    for label in sectoral.keys():
                        if self.canonical_sector(label) is None and label not in cache:
                            unknown.add(label)
                except Exception:
                    continue
        return unknown

    def build_gpt_mapping_prompt(self, unknown_labels: List[str]) -> str:
        numbered_labels = "\n".join(f"{i + 1}. {label}" for i, label in enumerate(unknown_labels))
        valid = ", ".join(sorted(VALID_SECTORS, key=lambda x: (x == OTHERS_SECTOR, x)))
        return (
            "Map each label to exactly one valid sector.\n"
            f"Valid sectors: {valid}\n"
            "Return ONLY a JSON object where keys are original labels and values are valid sectors.\n"
            "Prefer a real sector over Others whenever possible.\n\n"
            f"Labels:\n{numbered_labels}\n"
        )

    def call_gpt_for_mapping(self, unknown_labels: List[str]) -> Dict[str, str]:
        if not unknown_labels or self.openai_client is None:
            return {}

        prompt = self.build_gpt_mapping_prompt(unknown_labels)
        try:
            response = self.openai_client.chat.completions.create(
                model=SECTOR_GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a JSON-only sector classifier for Indian financial markets. "
                            "Return only the requested JSON object."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            raw_mapping = json.loads(content) if content else {}

            validated: Dict[str, str] = {}
            for label, mapped in (raw_mapping or {}).items():
                if not isinstance(label, str) or not isinstance(mapped, str):
                    continue
                if mapped in VALID_SECTORS:
                    validated[label] = mapped
                else:
                    validated[label] = STANDARD_SECTOR_LOWER.get(mapped.lower(), OTHERS_SECTOR)
            return validated
        except Exception as e:
            print(f"[Normalization] GPT mapping failed: {type(e).__name__}: {e}")
            return {}

    def prepare_sector_mapping_cache(self) -> Dict[str, str]:
        cache = self.load_mapping_cache()
        fixed = self.fix_cache_with_keywords(cache)
        if fixed:
            self.save_mapping_cache(cache)
            print(f"[Normalization] Fixed {fixed:,} cache entries locally.")

        unknown_labels = sorted(self.collect_unknown_sectors(cache))
        if not unknown_labels:
            return cache

        if self.openai_client is None:
            print(
                "[Normalization] OPENAI_API_KEY not found (or OpenAI SDK unavailable). "
                "Skipping API mapping and using local + cache normalization only."
            )
            return cache

        total_batches = (len(unknown_labels) + SECTOR_GPT_BATCH_SIZE - 1) // SECTOR_GPT_BATCH_SIZE
        print(f"[Normalization] {len(unknown_labels):,} labels unresolved locally -> {total_batches} GPT batches")

        for batch_num, i in enumerate(range(0, len(unknown_labels), SECTOR_GPT_BATCH_SIZE), start=1):
            batch = unknown_labels[i:i + SECTOR_GPT_BATCH_SIZE]
            print(f"[Normalization] Batch {batch_num}/{total_batches} ({len(batch)} labels)")
            mapped = self.call_gpt_for_mapping(batch)
            if not mapped:
                print("[Normalization] Empty mapping batch. Stopping further GPT mapping to avoid cache corruption.")
                break
            cache.update(mapped)
            for label in batch:
                cache.setdefault(label, OTHERS_SECTOR)
            self.save_mapping_cache(cache)

        return cache

    def normalize_sectoral_dict(self, sectoral: Dict[str, List[str]], cache: Dict[str, str]) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = defaultdict(list)
        for label, summaries in (sectoral or {}).items():
            target = self.canonical_sector(label)
            if target is None:
                target = cache.get(label, label)
            normalized[target].extend(list(summaries))
        return dict(normalized)

    def _extract_adani_tickers_from_text(self, text: str) -> Set[str]:
        if not text:
            return set()

        matches: Set[str] = set()
        if self.adani_enterprises_pattern.search(text):
            matches.add("ADANIENT")
        if self.adani_ports_pattern.search(text):
            matches.add("ADANIPORTS")
        return {ticker for ticker in matches if ticker in self.all_tickers}

    def _repair_direct_targets(self, original_ticker: str, summary: str) -> List[str]:
        repaired = self._extract_adani_tickers_from_text(summary)
        if repaired:
            return sorted(repaired)
        return [original_ticker]

    def build_final_csv(self) -> None:
        print("\n[Aggregation] Building final CSV from checkpoint file...")
        if not CHECKPOINT_FILE.exists():
            print("[Error] No checkpoint file found. Nothing to aggregate.")
            return

        mapping_cache = self.prepare_sector_mapping_cache()
        date_to_news = defaultdict(list)
        total_valid = 0

        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("skipped"):
                        continue
                    date_to_news[data["date"]].append(data["extracted_news"])
                    total_valid += 1
                except Exception:
                    continue

        import csv

        total_rows_written = 0
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=["date", "symbol", "direct_news", "sectoral_news", "global_news"])
            writer.writeheader()

            for date, news_list in date_to_news.items():
                tickers_data = defaultdict(lambda: {"direct": [], "sectoral": [], "global": []})

                for extracted_news in news_list:
                    for ticker, summaries in extracted_news.get("direct", {}).items():
                        for summary in summaries:
                            for repaired_ticker in self._repair_direct_targets(ticker, summary):
                                tickers_data[repaired_ticker]["direct"].append(summary)

                    normalized_sectoral = self.normalize_sectoral_dict(
                        extracted_news.get("sectoral", {}),
                        mapping_cache,
                    )
                    for sector_name, summaries in normalized_sectoral.items():
                        for summary in summaries:
                            for adani_ticker in self._extract_adani_tickers_from_text(summary):
                                tickers_data[adani_ticker]["sectoral"].append(summary)

                        if sector_name == OTHERS_SECTOR:
                            continue

                        matched_tickers = self.sector_to_tickers.get(sector_name, [])
                        if not matched_tickers:
                            for sect, keywords in SECTOR_KEYWORDS.items():
                                if any(keyword.lower() in sector_name.lower() for keyword in keywords) or sect.lower() in sector_name.lower():
                                    matched_tickers.extend(self.sector_to_tickers.get(sect, []))

                        for ticker in set(matched_tickers):
                            tickers_data[ticker]["sectoral"].extend(summaries)

                    for _, summaries in extracted_news.get("global", {}).items():
                        for ticker in self.all_tickers:
                            tickers_data[ticker]["global"].extend(summaries)

                for ticker, news in tickers_data.items():
                    writer.writerow(
                        {
                            "date": date,
                            "symbol": ticker,
                            "direct_news": " || ".join(news["direct"]) if news["direct"] else "",
                            "sectoral_news": " || ".join(news["sectoral"]) if news["sectoral"] else "",
                            "global_news": " || ".join(news["global"]) if news["global"] else "",
                        }
                    )
                    total_rows_written += 1

        print(f"[Aggregation] Final CSV saved to: {OUTPUT_CSV}")
        print(f"[Aggregation] Valid checkpoint rows read: {total_valid:,}")
        print(f"[Aggregation] Total unique Date/Ticker combinations written: {total_rows_written:,}")

    def retry_failed_rows(self, full_df: pd.DataFrame) -> int:
        failed_indices = self.load_failed_indices()
        failed_indices -= self.processed_indices

        if not failed_indices:
            return 0

        print(f"\n[Retry] Retrying {len(failed_indices):,} previously failed rows...")
        retry_df = full_df[full_df["original_index"].isin(failed_indices)].copy()
        retry_df = retry_df.set_index("original_index")
        if retry_df.empty:
            return 0

        success_count = 0
        with tqdm(total=len(retry_df), desc="Retrying failed rows", unit="article") as pbar:
            for start in range(0, len(retry_df), CHUNK_SIZE):
                if self.shutdown_event.is_set():
                    break
                chunk = retry_df.iloc[start:start + CHUNK_SIZE]
                success_count += self.process_failed_chunk(chunk, pbar)
                gc.collect()

        print(f"[Retry] Successfully recovered: {success_count:,} / {len(failed_indices):,}")
        return success_count

    def process_dataset(self, build_csv: bool = True) -> None:
        print("=== Qwen Financial NER Pipeline ===")
        print(f"Input: {INPUT_CSV}")
        print(f"Checkpoint: {CHECKPOINT_FILE}")
        print(f"Failed rows: {FAILED_ROWS_FILE}")
        print(f"Output CSV: {OUTPUT_CSV}")
        print(f"Model: {OLLAMA_MODEL}")
        print(f"Workers: {MAX_WORKERS}")
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"Retries: {MAX_RETRIES}")

        CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.processed_indices = self.load_processed_indices()

        total_rows = sum(1 for _ in open(INPUT_CSV, "r", encoding="utf-8")) - 1
        remaining_rows = max(total_rows - len(self.processed_indices), 0)
        print(f"Total articles: {total_rows:,}")
        print(f"Remaining to process: {remaining_rows:,}")

        if remaining_rows == 0:
            print("\n[Info] All articles already processed.")
            if build_csv:
                self.build_final_csv()
            else:
                print("[Info] Skipping CSV build (mode = NER only).")
            return

        full_df = pd.read_csv(INPUT_CSV, low_memory=False)
        full_df = full_df.iloc[::-1].reset_index(drop=False)
        full_df.rename(columns={"index": "original_index"}, inplace=True)

        def handle_sigint(_signum, _frame):
            if self.shutdown_event.is_set():
                print("\n[Ctrl+C] Force exit! Progress already saved.")
                os._exit(1)
            print("\n\n[Ctrl+C] Graceful shutdown... finishing in-flight rows...")
            print("[Ctrl+C] Press Ctrl+C again to force exit immediately.")
            self.shutdown_event.set()

        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, handle_sigint)

        total_success = 0
        interrupted = False
        print(f"\n[Phase 1] Processing remaining {remaining_rows:,} rows...")

        with tqdm(
            total=total_rows,
            desc="Processing articles (end->start)",
            initial=min(len(self.processed_indices), total_rows),
            unit="article",
        ) as pbar:
            for start in range(0, len(full_df), CHUNK_SIZE):
                if self.shutdown_event.is_set():
                    interrupted = True
                    break
                chunk = full_df.iloc[start:start + CHUNK_SIZE].set_index("original_index")
                total_success += self.process_chunk(chunk, pbar)
                gc.collect()

        if interrupted:
            print(f"\n[Interrupted] Stopped gracefully after processing {total_success:,} articles this session")
            print("[Interrupted] Re-run to continue from where you left off.")
        else:
            print(f"\n[Phase 1 Complete] Successfully processed: {total_success:,}")

        if not interrupted:
            self.processed_indices = self.load_processed_indices()
            self.retry_failed_rows(full_df)

        if FAILED_ROWS_FILE.exists():
            failed_count = sum(1 for line in open(FAILED_ROWS_FILE, "r", encoding="utf-8") if line.strip())
            if failed_count > 0:
                print(f"Failed rows remaining: {failed_count:,} (see {FAILED_ROWS_FILE})")

        if build_csv:
            self.build_final_csv()
        else:
            print("[Info] Skipping CSV build (mode = NER only).")

        signal.signal(signal.SIGINT, previous_handler)
        gc.collect()
        print(f"\nPipeline {'stopped' if interrupted else 'completed'}!")


def show_menu() -> int:
    """
    Startup mode menu:
      1 -> NER only
      2 -> CSV only
      3 -> Full pipeline
    """
    print()
    print("=" * 55)
    print("  Financial NER Pipeline - Mode Selection")
    print("=" * 55)
    print("  1)  NER only      - process articles -> raw_responses.jsonl")
    print("  2)  CSV only      - aggregate raw_responses.jsonl -> CSV")
    print("  3)  Full pipeline - NER + CSV")
    print("=" * 55)

    while True:
        try:
            choice = input("  Select mode [1/2/3]: ").strip()
            if choice in ("1", "2", "3"):
                return int(choice)
            print("  Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            raise SystemExit(0)


def run(mode: int) -> None:
    pipeline = QwenFinancialNERPipeline()

    if mode == 1:
        print("\n[Mode 1] NER only - will produce raw_responses.jsonl")
        pipeline.process_dataset(build_csv=False)
    elif mode == 2:
        print("\n[Mode 2] CSV only - aggregating existing raw_responses.jsonl...")
        pipeline.processed_indices = pipeline.load_processed_indices()
        pipeline.build_final_csv()
    else:
        print("\n[Mode 3] Full pipeline - NER + CSV")
        pipeline.process_dataset(build_csv=True)


if __name__ == "__main__":
    selected_mode = show_menu()
    run(selected_mode)
