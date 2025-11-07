# Clinical-NER-Engine-
Clinical NER Engine 

A **Streamlit** app for running **clinical Named Entity Recognition (NER)** using multiple Hugging Face models.
It provides clean dashboards, color highlights, and model comparison tools **without any gold-standard labels**.

## What It Does

This app is designed for **exploratory evaluation** of NER models on clinical text. You can:

- **Upload TXT or PDF clinical notes** or paste sample text  
- **Run multiple models** at once (e.g., `d4data/biomedical-ner-all`, `dslim/bert-base-NER`)  
- **Highlight detected entities** directly in the text  
- **Search** by model, label, or confidence  
- **Compare models** using overlap and diversity metrics  
- **View dashboards** that summarize model behavior  
- **Compute consensus** across models (majority or unanimous)  
- **Measure inference speed** for each model  

Everything runs locally and stores results in a small SQLite database (`ner_index.db`).

---

## How It Works

1. **Load Models**  
In the sidebar, list one or more Hugging Face NER models (each on a new line):  
d4data/biomedical-ner-all
dslim/bert-base-NER
Jean-Baptiste/camembert-ner


2. **Upload or Paste Text**  
Upload `.txt` or `.pdf` notes or paste clinical text (e.g., a discharge summary).

3. **Run NER**  
Each model runs once per document.  
Every entity with a confidence score ≥ your chosen threshold is stored in a local SQLite database.

4. **Explore Tabs**  
View highlights, search results, model comparisons.

---

## 📊 Metrics Explained (With Examples)

| Metric | What It Measures | Example |
|--------|------------------|----------|
| **Dice Similarity** | Agreement between two models (higher = more overlap). | Model A finds `{aspirin, metoprolol}`; Model B finds `{aspirin, furosemide}` → Dice = `2×1 / (2+2)` = **0.5**. |
| **Jaccard Similarity** | Another overlap score using union instead of total count. | Same example → Jaccard = `1 / 3 ≈ 0.33`. |
| **Per-Label Overlap** | Overlap but computed per entity type (e.g., only `DRUG`). | Helps see which labels align most across models. |
| **Type–Token Ratio (TTR)** | Diversity = unique entities ÷ total entities. | 10 entities, 5 unique → TTR = 0.5. Higher = more variety. |
| **Label Entropy** | How balanced predictions are across labels. | If 90% are `DRUG`, entropy is low (imbalanced). |
| **Span Length (mean, p50, p90)** | Avg characters per entity. | “asthma” = short; “acute lymphoblastic leukemia” = long. |
| **Score Stats (mean, p50, p90)** | Confidence distribution per label. | Shows which entities are more confidently predicted. |
| **Document Density** | Unique `(label, term)` per 1 000 chars. | If 4 unique entities per 1 000 chars → density = 4. |
| **Threshold Sweep** | Compares metrics at different confidence cutoffs. | At 0.3 → 200 entities; at 0.7 → 90 entities. |
| **Consensus (Majority / Unanimous)** | Entities agreed on by most or all models. | 2 of 3 models find “asthma” → majority; 3 of 3 → unanimous. |
| **Latency (p50 / p90 / p99)** | Inference time percentiles (ms). | p50 = median speed; p99 = slowest runs. |

These metrics describe **agreement**, **diversity**, and **coverage** — useful when no human-annotated (“gold”) data exists.

---

## 🖥️ Tabs Overview

### 🧪 Process
- Upload `.txt` / `.pdf` files or paste text.  
- Choose models and set a confidence threshold.  
- Click **Run NER** to extract and save entities.  
- View entity counts and highlighted text.

### 🖍️ Highlights
- Pick a note and model.  
- Optionally filter by labels.  
- See full text with **color-coded spans** and **label + score** tooltips.

### 🔎 Search
- Search note content.  
- Filter by **model**, **label**, or **score**.  
- Export results as CSV.

