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

