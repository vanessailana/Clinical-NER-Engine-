# app.py — Clinical NER (Single-Pass, No Aggregation, No Polarity)
# ----------------------------------------------------------------
# - One full inference per document (no chunking)
# - aggregation_strategy="none" (for token-classifiers)
# - LLM path available via "LLM:<hf_model_id>" entries
# - RECALL BOOSTS: normalize, expand dosage tails, merge same-label tokens
# - Overlap-aware highlighting (renders ALL overlapping entities)
# - Persist-all; confidence is a VIEW-TIME filter only
# - Duplicate suppression: drop exact dupes and nested same-label spans
# - Per-file×model CSV export + combined CSV export

from __future__ import annotations

# ------------------------------ Env & Imports ---------------------------------
import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import re
import json
import sqlite3
import time
import traceback
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import streamlit as st

# Optional deps (safe to miss on CPU-only hosts)
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# Hugging Face (local) — no API key required for public models
try:
    from transformers import (
        AutoModelForTokenClassification,
        AutoModelForCausalLM,
        AutoTokenizer,
        pipeline as hf_pipeline,
    )
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# ------------------------------ Page / Styles ---------------------------------
st.set_page_config(page_title="Clinical NER — Single-Pass (No-Gold)", layout="wide", page_icon="🩺")

st.markdown(
    """
<style>
.main .block-container { padding-top: .8rem; padding-bottom: 2.2rem; }
.ner-span { padding: 2px 6px; border-radius: 6px; }
.rich-text { max-width: 1100px; line-height: 1.55; }
.section-label { opacity: 0.8; font-size: 0.9rem; }
.legend-item { display:flex; align-items:center; gap:8px; margin:4px 12px 4px 0; }
.legend-box { width:16px; height:16px; border-radius:4px; display:inline-block; }
.stMetric > div { justify-content: center; }
.multi-label { border: 1px dashed rgba(0,0,0,.25); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Clinical NER")

# -------------------------------- Constants -----------------------------------
ENTITY_COLORS = [
    "#f9c74f", "#90be6d", "#f94144", "#577590", "#43aa8b",
    "#f9844a", "#4d908e", "#277da1", "#e76f51", "#8ab17d",
    "#b56576", "#6d597a", "#355070", "#43bccd", "#c2c5aa"
]
DB_PATH = "ner_index.db"
EXPORT_DIR = "exports"

# Ensure export dir exists
os.makedirs(EXPORT_DIR, exist_ok=True)

# Expanded dosage tail: add IU, meq, %, tabs/caps, ER/SR, qXh, qXd, PRN variants
DOSAGE_TAIL_RE = re.compile(
    r"""^(
         \s*(\d+(\.\d+)?)\s*
         (mg|mcg|µg|ug|g|kg|mL|ml|L|IU|units|meq|%)\s*
         (/?\s*(tab(s)?|cap(s)?|spray(s)?|puff(s)?))?\s*
         (/?\s*(ER|SR|XR|IR))?\s*
         (/?\s*(q\d+h|q\d+d|BID|TID|QID|qHS|PRN|prn|daily|OD))?
       )""",
    re.IGNORECASE | re.VERBOSE,
)

LLM_JSON_INSTR = (
    "Extract all clinical entities from the note. "
    "Use labels from this set when applicable: {labels}. "
    "Return EXACT character offsets over the ORIGINAL text.\n\n"
    "Respond ONLY with a JSON array of objects of the form:\n"
    '[{"label": "DRUG|DISEASE|SYMPTOM|ANATOMY|TEST|PROCEDURE", '
    '"start": <int>, "end": <int>, "text": "<exact substring>"}]\n\n'
    "Note:\n{note}"
)

DEFAULT_LABEL_SET = ["DRUG", "DISEASE", "SYMPTOM", "ANATOMY", "TEST", "PROCEDURE"]

# -------------------------------- Utilities -----------------------------------
def color_for_label(label: str) -> str:
    return ENTITY_COLORS[abs(hash(label)) % len(ENTITY_COLORS)]

def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

def normalize_text_for_inference(text: str) -> str:
    if not text:
        return text
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\r\n?", "\n", t)
    return t

def dedup_order(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s.strip())
    return s[:120] or "note"

# --- Span utilities to avoid duplicate highlights ---
def merge_adjacent_same_label(ents: List[Dict[str, Any]], text: str, max_gap: int = 1) -> List[Dict[str, Any]]:
    if not ents:
        return []
    ents = sorted(ents, key=lambda e: (int(e["start"]), int(e["end"])))
    merged: List[Dict[str, Any]] = []
    cur = ents[0].copy()
    for nxt in ents[1:]:
        if (str(nxt["entity_group"]) == str(cur["entity_group"])
            and 0 <= int(nxt["start"]) - int(cur["end"]) <= max_gap):
            mid = text[int(cur["end"]):int(nxt["start"])]
            if re.fullmatch(r"[\s\-\–\—/(),]*", mid or ""):
                cur["end"] = int(nxt["end"])
                cur["score"] = max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0)))
                cur["word"] = text[int(cur["start"]):int(cur["end"])]
                continue
        merged.append(cur)
        cur = nxt.copy()
    merged.append(cur)
    return merged

def dedup_spans_exact(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[int,int,str]] = set()
    out: List[Dict[str, Any]] = []
    for e in ents:
        key = (int(e["start"]), int(e["end"]), str(e.get("entity_group") or e.get("label") or ""))
        if key in seen:
            continue
        seen.add(key)
        if "entity_group" not in e and "label" in e:
            e = dict(e); e["entity_group"] = e.pop("label")
        out.append(e)
    return out

def suppress_nested_same_label(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ents:
        return []
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for e in ents:
        lbl = str(e.get("entity_group") or e.get("label") or "")
        e = dict(e); e["entity_group"] = lbl
        by_label.setdefault(lbl, []).append(e)

    kept: List[Dict[str, Any]] = []
    for label, ls in by_label.items():
        ls = sorted(ls, key=lambda e: (int(e["start"]), -int(e["end"]), -float(e.get("score",0.0))))
        selected: List[Dict[str, Any]] = []
        for e in ls:
            s, ed = int(e["start"]), int(e["end"])
            contained = any(int(k["start"]) <= s and ed <= int(k["end"]) for k in selected)
            if not contained:
                selected.append(e)
        kept.extend(selected)
    return dedup_spans_exact(kept)

def build_segments_with_overlaps(text: str, ents: List[Dict[str, Any]]) -> List[Tuple[int,int,List[Dict[str,Any]]]]:
    n = len(text)
    if not ents or n == 0:
        return [(0, n, [])]
    spans = [(max(0,int(e["start"])), max(0,min(n,int(e["end"])))) for e in ents if int(e["end"])>int(e["start"])]
    bps = sorted({0, n, *[s for s,_ in spans], *[e for _,e in spans]})
    segs: List[Tuple[int,int,List[Dict[str,Any]]]] = []
    for a, b in zip(bps, bps[1:]):
        covering = []
        for e in ents:
            s, e_ = int(e["start"]), int(e["end"])
            if s < b and e_ > a:
                covering.append(e)
        segs.append((a, b, covering))
    return segs

def highlight_text_overlap_aware(text: str, ents: List[Dict[str, Any]]) -> str:
    if not text:
        return "<div class='rich-text'>(empty)</div>"
    if not ents:
        return f"<div class='rich-text'>{escape_html(text)}</div>"

    ents = sorted(ents, key=lambda e: (int(e["start"]), int(e["end"])))
    segs = build_segments_with_overlaps(text, ents)

    out_parts: List[str] = []
    for a, b, covering in segs:
        raw = escape_html(text[a:b])
        if not covering:
            out_parts.append(raw)
            continue
        covering_sorted = sorted(covering, key=lambda e: float(e.get("score",0.0)), reverse=True)
        top = covering_sorted[0]
        labels = dedup_order([str(c.get("entity_group","?")) for c in covering_sorted])
        scores = [f"{c.get('entity_group','?')}={float(c.get('score',0.0)):.2f}" for c in covering_sorted]
        tip = f"{' | '.join(labels)} — " + ", ".join(scores)
        col = color_for_label(str(top.get("entity_group","?")))
        klass = "ner-span multi-label" if len(labels) > 1 else "ner-span"
        out_parts.append(f"<span class='{klass}' title='{escape_html(tip)}' style='background:{col}'>{raw}</span>")

    return f"<div class='rich-text'>{''.join(out_parts)}</div>"

def _device_index() -> int:
    return 0 if (HAS_TORCH and torch.cuda.is_available()) else -1

def canonical_model_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def pct_list(vals: Sequence[float], q: float) -> Optional[float]:
    if not vals:
        return None
    vs = sorted(vals)
    idx = int(max(0, min(len(vs) - 1, (len(vs) - 1) * q)))
    return float(vs[idx])

# --------------------------------- DB Helpers ---------------------------------
@st.cache_resource(show_spinner=False)
def get_conn(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents(
          id INTEGER PRIMARY KEY,
          bundle TEXT,
          name TEXT,
          text TEXT
        );
        CREATE TABLE IF NOT EXISTS entities(
          id INTEGER PRIMARY KEY,
          doc_id INTEGER,
          model TEXT,
          label TEXT,
          word TEXT,
          score REAL,
          start INTEGER,
          end INTEGER,
          FOREIGN KEY(doc_id) REFERENCES documents(id)
        );
        CREATE TABLE IF NOT EXISTS perf_events(
          id INTEGER PRIMARY KEY,
          ts REAL,
          model TEXT,
          doc_id INTEGER,
          phase TEXT,
          ms REAL
        );
        """
    )
    conn.commit()
    return conn

CONN = get_conn()

def sql_df(sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, CONN, params=params or [])

def sql_exec(sql: str, params: Sequence[Any] | None = None) -> None:
    CONN.execute(sql, params or [])
    CONN.commit()

def sql_many(sql: str, rows: Sequence[Sequence[Any]]) -> None:
    CONN.executemany(sql, rows)
    CONN.commit()

# ------------------------------ LLM Helpers -----------------------------------
def is_llm_entry(model_name: str) -> bool:
    """Detect 'LLM:<hf_id>' prefix to route to the LLM JSON extractor."""
    return bool(re.match(r"^\s*LLM\s*:\s*", model_name))

def llm_model_id(model_name: str) -> str:
    """Strip 'LLM:' prefix and return the Hugging Face model id."""
    return re.sub(r"^\s*LLM\s*:\s*", "", model_name).strip()

def extract_json_list(s: str) -> List[Dict[str, Any]]:
    """
    Robustly extract a JSON array from LLM text. Handles ```json blocks or stray text.
    """
    if not s:
        return []
    # Prefer fenced code blocks
    fence = re.search(r"```json\s*(\[.*?\])\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1)
    else:
        # Fallback: first top-level array
        arr = re.search(r"\[\s*{.*}\s*\]", s, flags=re.DOTALL)
        if arr:
            s = arr.group(0)
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []

def llm_ner_pipeline_factory(hf_id: str):
    """
    Returns a callable(text) -> list[dict(label, word, score, start, end)]
    using a causal LLM with a JSON extraction prompt.
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers not available. pip install transformers accelerate")

    try:
        tok = AutoTokenizer.from_pretrained(hf_id)
        mdl = AutoModelForCausalLM.from_pretrained(hf_id)
        gen = hf_pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            device=_device_index(),
        )
    except Exception as ex:
        raise RuntimeError(f"Failed to load LLM '{hf_id}': {ex}")

    labels_str = ", ".join(DEFAULT_LABEL_SET)

    def _call(text: str) -> List[Dict[str, Any]]:
        note = text if isinstance(text, str) else str(text or "")
        prompt = LLM_JSON_INSTR.format(labels=labels_str, note=note)
        try:
            out = gen(prompt, max_new_tokens=512, do_sample=False, temperature=0.0)[0]["generated_text"]
        except Exception as ex:
            # Generate may include the prompt; if so, keep only completion tail
            out = f"[]  /* generation error: {ex} */"

        items = extract_json_list(out)
        ents: List[Dict[str, Any]] = []

        for it in items:
            label = str(it.get("label", "")).strip() or "ENTITY"
            text_piece = str(it.get("text", "") or it.get("word", "") or "")
            start = it.get("start", None)
            end = it.get("end", None)

            # If offsets missing but we have text, try to locate first occurrence
            if (start is None or end is None) and text_piece:
                idx = note.find(text_piece)
                if idx >= 0:
                    start, end = idx, idx + len(text_piece)

            # Validate offsets
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(note):
                ents.append({
                    "entity_group": label,
                    "word": note[start:end],
                    "score": 0.99,   # pseudo-confidence for LLM path
                    "start": start,
                    "end": end,
                })
            # else: skip invalid rows silently

        return ents

    return _call

# ------------------------------ Models / Pipelines ----------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str):
    """
    Returns a callable pipeline with signature: pipeline(text) -> list[dict(entity|label, word, score, start, end)]
    Supports:
      • HF token-classification models (aggregation_strategy=None)
      • LLM JSON extractor via entries like 'LLM:<hf_model_id>'
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Transformers is not available. Please install it: pip install transformers accelerate")

    # LLM route (prompted JSON extractor)
    if is_llm_entry(model_name):
        return llm_ner_pipeline_factory(llm_model_id(model_name))

    # Token-classifier route
    try:
        return hf_pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="none",
            device=_device_index(),
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForTokenClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
        return hf_pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy="none",
            device=_device_index(),
        )

def safe_run_pipe(pipe, text: str) -> Tuple[List[Dict[str, Any]], float, bool, Optional[str]]:
    t0 = time.perf_counter()
    try:
        ents = pipe(text)
        ok, err = True, None
    except Exception as ex:
        ents, ok, err = [], False, "".join(traceback.format_exception_only(type(ex), ex)).strip()
    ms = (time.perf_counter() - t0) * 1000.0
    return ents, ms, ok, err

def expand_dosage_tail(text: str, ent: Dict[str, Any]) -> Dict[str, Any]:
    tail = text[int(ent["end"]) : int(ent["end"]) + 60]
    m = DOSAGE_TAIL_RE.match(tail)
    if m:
        ent = dict(ent)
        ent["end"] = int(ent["end"]) + m.end()
        ent["word"] = text[int(ent["start"]) : int(ent["end"])]
    return ent

def run_ner_single_pass(pipe, text: str) -> List[Dict[str, Any]]:
    """
    Single full-document inference with MANDATORY recall boosts:
      - normalize text
      - expand dosage tails
      - merge adjacent same-label tokens
    Persist all spans (no confidence filtering here).
    """
    text_in = normalize_text_for_inference(text)
    outs, _, ok, _ = safe_run_pipe(pipe, text_in)
    if not ok or not outs:
        return []

    # Normalize outputs to a common schema with entity_group
    results: List[Dict[str, Any]] = []
    for o in outs:
        ent = {
            "entity_group": o.get("entity_group") or o.get("entity") or o.get("label") or "",
            "word": o.get("word", ""),
            "score": float(o.get("score", 0.0)),
            "start": int(o.get("start", 0)),
            "end": int(o.get("end", 0)),
        }
        # ALWAYS expand dosage tails (clinical)
        ent = expand_dosage_tail(text_in, ent)
        results.append(ent)

    # ALWAYS merge contiguous tokens of same label
    results = merge_adjacent_same_label(results, text_in, max_gap=1)

    # Suppress exact dups and nested same-label spans
    results = suppress_nested_same_label(dedup_spans_exact(results))

    # Map words back to original (pre-normalization) text for rendering
    for e in results:
        s, ed = int(e["start"]), int(e["end"])
        s = max(0, min(len(text), s))
        ed = max(0, min(len(text), ed))
        e["word"] = text[s:ed]
        e["start"], e["end"] = s, ed

    return results

# -------------------------------- Sidebar -------------------------------------
def sidebar_settings():
    st.sidebar.subheader("Settings")

    default_models = "\n".join([
        # HF token-classification models (download once; no API key required)
        "d4data/biomedical-ner-all",
        "dslim/bert-base-NER",
        "kamalkraj/BioClinicalBERT-NER"
        "Jean-Baptiste/camembert-ner",
        "emilyalsentzer/Bio_ClinicalBERT",
        "Helios9/BIOMed_NER",
        "HUMADEX/english_medical_ner"
       
    ])
    models_raw = st.sidebar.text_area(
        "Models (one per line)",
        default_models,
        height=140,
        key="sb_models",
        help="Token-classifiers run directly. For a prompted clinical LLM extractor, write LLM:<hf_model_id> (e.g., LLM:epfl-llm/meditron-7b).",
    )
    model_list = dedup_order([canonical_model_name(m) for m in models_raw.splitlines() if m.strip()])
    st.sidebar.caption(f"👀 Distinct models loaded: {len(model_list)}")

    conf = st.sidebar.slider("Min confidence (view only)", 0.0, 1.0, 0.50, 0.01, key="sb_conf")
    enable_pdf = st.sidebar.checkbox("Enable PDF parsing", value=HAS_PYPDF, key="sb_pdf")
    st.sidebar.caption(f"CUDA available: {torch.cuda.is_available() if HAS_TORCH else False}")
    if not HAS_TRANSFORMERS:
        st.sidebar.error("Transformers not installed. Run: pip install transformers accelerate")

    return {
        "models": model_list,
        "conf": float(conf),      # view-time threshold only
        "pdf": bool(enable_pdf),
    }

# -------------------------------- Tabs ----------------------------------------
def counters():
    n_docs = CONN.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_ents = CONN.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    n_models = CONN.execute("SELECT COUNT(DISTINCT model) FROM entities").fetchone()[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", n_docs)
    c2.metric("Entities", n_ents)
    c3.metric("Distinct models", n_models)

def tab_process_ui(cfg: Dict[str, Any]):
    st.markdown("#### Process notes (single-pass pipeline)")
    c1, c2 = st.columns([1, 1])
    with c1:
        uploads = st.file_uploader("Upload files (.txt / .pdf)", type=["txt", "pdf"], accept_multiple_files=True, key="proc_uploader")
    with c2:
        demo_text = st.text_area(
            "Demo / paste text",
            value=(
                "The patient reported no recurrence of palpitations at follow-up 6 months after the ablation. "
                "Current meds include metoprolol 25 mg BID. Denies chest pain or dyspnea. Possible PVCs on prior Holter."
            ),
            height=160,
            key="proc_demo_text",
        )

    c3, c4, c5 = st.columns([1, 1, 2])
    with c3:
        run_clicked = st.button("▶ Run NER", type="primary", key="proc_run")
    with c4:
        clear_clicked = st.button("🧹 Clear index", key="proc_clear")
    with c5:
        st.caption("Single-pass; mandatory recall boosts; overlap-aware highlights; nested same-label dupes removed. CSVs are saved automatically.")

    if clear_clicked:
        sql_exec("DELETE FROM entities")
        sql_exec("DELETE FROM documents")
        sql_exec("DELETE FROM perf_events")
        st.success("Index cleared.")

    if not run_clicked:
        counters()
        return

    if not cfg["models"]:
        st.error("Add at least one model in the sidebar.")
        counters()
        return

    files = uploads or []
    if not files and demo_text.strip():
        class _VirtualFile:
            name = "demo_note.txt"
            def read(self_inner): return demo_text.encode("utf-8")
        files = [_VirtualFile()]

    if not files:
        st.warning("Nothing to process. Upload a file or keep demo text non-empty.")
        counters()
        return

    # Collect all rows for a combined CSV at the end
    all_rows_for_csv: List[Dict[str, Any]] = []

    prog_files = st.progress(0.0)
    for fidx, up in enumerate(files):
        name = getattr(up, "name", "note.txt")
        suffix = name.lower().split(".")[-1]

        if suffix == "pdf" and cfg["pdf"] and HAS_PYPDF:
            try:
                raw = up.read()
                reader = PdfReader(io.BytesIO(raw))
                text = "\n".join([(p.extract_text() or "") for p in reader.pages])
            except Exception as ex:
                st.warning(f"PDF parse failed for {name}: {ex}")
                text = ""
        else:
            try:
                text = up.read().decode("utf-8", "ignore")
            except Exception:
                text = ""

        prog_files.progress((fidx + 1) / max(1, len(files)))

        # Insert document
        cur = CONN.cursor()
        cur.execute("INSERT INTO documents(bundle, name, text) VALUES (?, ?, ?)", (None, name, text))
        doc_id = cur.lastrowid
        CONN.commit()

        st.markdown(f"##### {name}")
        st.caption(f"Running {len(cfg['models'])} model(s)…")

        base = sanitize_filename(name.rsplit("/", 1)[-1])

        for m in cfg["models"]:
            m_canon = canonical_model_name(m)
            m_slug = sanitize_filename(m_canon)
            with st.expander(f"Results — {name} — `{m_canon}`", expanded=False):
                mode = "LLM JSON extractor" if is_llm_entry(m_canon) else "token-classification (aggregation='none')"
                st.write(f"Loading `{m_canon}` — {mode}…")
                try:
                    pipe = load_pipeline(m_canon)
                except Exception as ex:
                    st.error(f"Failed to load `{m_canon}`: {ex}")
                    continue

                t0 = time.perf_counter()
                ents = run_ner_single_pass(pipe, text)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                sql_exec(
                    "INSERT INTO perf_events(ts, model, doc_id, phase, ms) VALUES (?, ?, ?, ?, ?)",
                    (time.time(), m_canon, doc_id, "infer", float(infer_ms)),
                )

                # Persist ALL entities (no filtering)
                if ents:
                    rows_db = [
                        (doc_id, m_canon, e["entity_group"], e["word"], float(e["score"]), int(e["start"]), int(e["end"]))
                        for e in ents
                    ]
                    sql_many(
                        "INSERT INTO entities(doc_id, model, label, word, score, start, end) VALUES (?,?,?,?,?,?,?)",
                        rows_db,
                    )

                # Prepare tabular rows (include filename & model)
                rows_tab = [{
                    "file_name": name,
                    "model": m_canon,
                    "label": e["entity_group"],
                    "word": e["word"],
                    "score": float(e["score"]),
                    "start": int(e["start"]),
                    "end": int(e["end"]),
                    "doc_id": int(doc_id)
                } for e in ents]

                # Append to combined export
                all_rows_for_csv.extend(rows_tab)

                # Save per-file×model CSV
                df_csv = pd.DataFrame(rows_tab)
                per_path = os.path.join(EXPORT_DIR, f"entities_{base}__{m_slug}.csv")
                try:
                    df_csv.to_csv(per_path, index=False)
                    st.success(f"Saved CSV: {per_path}")
                    st.download_button(
                        label="⬇️ Download this CSV",
                        data=df_csv.to_csv(index=False).encode("utf-8"),
                        file_name=os.path.basename(per_path),
                        mime="text/csv",
                        key=f"dl_{doc_id}_{m_slug}"
                    )
                except Exception as ex:
                    st.warning(f"Could not save CSV for {m_canon}: {ex}")

                # View-time filter + duplicate suppression for highlights/table
                ents_view = [e for e in ents if float(e.get("score", 0.0)) >= float(cfg["conf"])]
                ents_view = suppress_nested_same_label(dedup_spans_exact(ents_view))

                st.write(
                    f"Detected **{len(ents)}** entities "
                    f"(**{len(ents_view)}** shown at ≥ {cfg['conf']:.2f}). "
                    f"Inference: {infer_ms:.1f} ms"
                )

                if text.strip():
                    st.markdown(
                        highlight_text_overlap_aware(text, ents_view),
                        unsafe_allow_html=True
                    )
                if ents_view:
                    df_show = pd.DataFrame(
                        [{
                            "label": e["entity_group"],
                            "word": e["word"],
                            "score": round(float(e["score"]), 4),
                            "start": int(e["start"]),
                            "end": int(e["end"]),
                        } for e in ents_view]
                    )
                    st.dataframe(
                        df_show.sort_values(["label", "score", "word"], ascending=[True, False, True]),
                        use_container_width=True
                    )

    # Combined export for the whole run
    if all_rows_for_csv:
        st.markdown("#### 📦 Combined CSV for this run")
        df_all = pd.DataFrame(all_rows_for_csv)
        combined_path = os.path.join(EXPORT_DIR, f"entities_all_{int(time.time())}.csv")
        try:
            df_all.to_csv(combined_path, index=False)
            st.success(f"Saved combined CSV: {combined_path}")
        except Exception as ex:
            st.warning(f"Could not save combined CSV: {ex}")

        st.download_button(
            label="⬇️ Download combined CSV (all files & models in this run)",
            data=df_all.to_csv(index=False).encode("utf-8"),
            file_name=os.path.basename(combined_path),
            mime="text/csv",
            key="dl_all_csv"
        )

    counters()

def tab_highlight_ui():
    st.markdown("#### Full-text highlights")
    docs = sql_df("SELECT id, name FROM documents ORDER BY id DESC")
    models_all = [r[0] for r in CONN.execute("SELECT DISTINCT model FROM entities ORDER BY model").fetchall()]
    labels_all = [r[0] for r in CONN.execute("SELECT DISTINCT label FROM entities ORDER BY label").fetchall()]
    if docs.empty or not models_all:
        st.info("Process at least one note in the **Process** tab.")
        return

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        doc_sel_name = st.selectbox("Document", options=docs["name"].tolist(), index=0, key="hl_doc_sel")
    with c2:
        model_sel = st.selectbox("Model", options=models_all, index=0, key="hl_model_sel")
    with c3:
        label_filter = st.multiselect("Show only labels (optional)", options=labels_all, default=[], key="hl_label_filter")
    with c4:
        min_score = st.slider("Min entity score", 0.0, 1.0, st.session_state.get("sb_conf", 0.5), 0.01, key="hl_min_score")

    row = docs[docs["name"] == doc_sel_name].iloc[0]
    doc_id = int(row["id"])
    doc_text = CONN.execute("SELECT text FROM documents WHERE id=?", (doc_id,)).fetchone()[0]

    if label_filter:
        df_ents = sql_df(
            "SELECT label AS entity_group, word, score, start, end FROM entities "
            f"WHERE doc_id=? AND model=? AND label IN ({','.join('?' * len(label_filter))}) AND score>=? "
            "ORDER BY start, end",
            [doc_id, model_sel, *label_filter, float(min_score)],
        )
    else:
        df_ents = sql_df(
            "SELECT label AS entity_group, word, score, start, end FROM entities "
            "WHERE doc_id=? AND model=? AND score>=? ORDER BY start, end",
            [doc_id, model_sel, float(min_score)],
        )
    ents = df_ents.to_dict(orient="records")
    ents = suppress_nested_same_label(dedup_spans_exact(ents))

    st.markdown("**Legend**")
    if len(ents) == 0:
        st.caption("No entities found for this selection (try lowering Min entity score).")
    else:
        used_labels = sorted({e["entity_group"] for e in ents})
        st.markdown(
            "".join(
                f"<span class='legend-item'><span class='legend-box' style='background:{color_for_label(lb)}'></span>"
                f"<span class='section-label'>{escape_html(lb)}</span></span>"
                for lb in used_labels
            ),
            unsafe_allow_html=True,
        )

    st.markdown("**Highlighted note**")
    st.markdown(highlight_text_overlap_aware(doc_text, ents), unsafe_allow_html=True)

def tab_search_ui():
    st.markdown("#### Search entities")
    models_all = [r[0] for r in CONN.execute("SELECT DISTINCT model FROM entities ORDER BY model").fetchall()]
    labels_all = [r[0] for r in CONN.execute("SELECT DISTINCT label FROM entities ORDER BY label").fetchall()]

    q = st.text_input("Document text contains", "", key="search_text")
    sel_models = st.multiselect("Models", options=models_all, default=models_all, key="search_models")
    sel_labels = st.multiselect("Labels", options=labels_all, default=[], key="search_labels")
    min_score = st.slider("Min entity score", 0.0, 1.0, 0.5, 0.01, key="search_minscore")

    if st.button("Run Search", key="search_go", type="primary"):
        sql = """
            SELECT d.name,
                   e.model,
                   e.label AS entity_group,
                   e.word,
                   e.score,
                   e.start,
                   e.end
            FROM entities e
            JOIN documents d ON d.id = e.doc_id
            WHERE 1=1
        """
        params: List[Any] = []
        if q.strip():
            sql += " AND d.text LIKE ?"; params.append(f"%{q.strip()}%")
        if sel_models:
            sql += " AND e.model IN ({})".format(",".join("?" * len(sel_models))); params += sel_models
        if sel_labels:
            sql += " AND e.label IN ({})".format(",".join("?" * len(sel_labels))); params += sel_labels
        sql += " AND e.score >= ?"; params.append(float(min_score))
        sql += " ORDER BY e.model, e.label, e.word"
        df = sql_df(sql, params)
        if df.empty:
            st.info("No matches.")
        else:
            ents_list = df.to_dict(orient="records")
            ents_list = suppress_nested_same_label(dedup_spans_exact(ents_list))
            df_show = pd.DataFrame(ents_list).rename(columns={"name": "file_name"})
            st.dataframe(df_show, use_container_width=True)
            st.download_button(
                "⬇️ Download search results (CSV)",
                df_show.to_csv(index=False).encode("utf-8"),
                "ner_search_results.csv",
                "text/csv",
                key="dl_search_csv"
            )

def tab_analytics_ui():
    st.markdown("#### Analytics")
    if CONN.execute("SELECT COUNT(*) FROM perf_events").fetchone()[0] == 0:
        st.info("Process at least one note in the **Process** tab to see analytics.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Inference performance (ms)**")
        df_perf = sql_df(
            "SELECT model, ms FROM perf_events WHERE phase='infer'"
        )
        perf_agg = df_perf.groupby("model")["ms"].agg(["mean", "median", "count"]).sort_values("mean")
        st.dataframe(perf_agg, use_container_width=True)

    with c2:
        st.markdown("**Entity counts by label & model**")
        df_counts = sql_df(
            "SELECT model, label, COUNT(*) AS count FROM entities GROUP BY 1, 2 ORDER BY 1, 2"
        )
        counts_pivot = df_counts.pivot_table(index="label", columns="model", values="count").fillna(0).astype(int)
        st.dataframe(counts_pivot, use_container_width=True)

    st.markdown("**Confidence score distributions (P50 / P90 / Max)**")
    df_scores = sql_df("SELECT model, label, score FROM entities")
    score_agg = df_scores.groupby(["model", "label"])["score"].agg([
        "count",
        lambda x: pct_list(list(x), 0.5),
        lambda x: pct_list(list(x), 0.9),
        "max",
    ]).rename(columns={"<lambda_0>": "median", "<lambda_1>": "p90"})
    st.dataframe(score_agg, use_container_width=True)


# -------------------------------- Main ----------------------------------------
def main():
    cfg = sidebar_settings()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Process", "Highlight", "Search", "Analytics"]
    )
    with tab1:
        tab_process_ui(cfg)
    with tab2:
        tab_highlight_ui()
    with tab3:
        tab_search_ui()
    with tab4:
        tab_analytics_ui()

if __name__ == "__main__":
    main()