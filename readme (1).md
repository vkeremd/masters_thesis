# Voices of Outrage: An Event Study on the Shift in Online Discourse following Notable Police Violence Deaths in the United States using a Novel Transformer-Based BERTweet Fine-Tuned Model Approach

This repository houses the code for my Master’s thesis.  The project builds a complete machine‑learning pipeline that **classifies xx million tweets for racial‑justice–related content (“perceived injustice” and "systemic racism") and maps results to U.S. counties** to be used in an event study setting.


The pipeline spans **data labelling (GPT‑4), fine‑tuning a BERTweet model, large‑scale inference, geolocation, and aggregation**.  Each step is modular so you can reproduce or extend individual stages.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup & Dependencies](#setup--dependencies)
- [Data](#data)
- [Running the Pipeline](#running-the-pipeline)
  - [Phase 1 – Training‑Data Creation](#phase-1--training-data-creation)
  - [Phase 2 – Model Training](#phase-2--model-training)
  - [Phase 3 – Full Classification](#phase-3--full-classification)
  - [Phase 4 – Location + County Mapping](#phase-4--location--county-mapping)
  - [Phase 5 – Analysis & Validation](#phase-5--analysis--validation)
- [Reproducibility](#reproducibility)
- [Contact](#contact)

---

## Overview

The project answers a simple—but data‑intensive—question: **How often do tweets mention racial injustice and where are those conversations happening?**

1. **GPT‑4‑powered labelling** creates a high‑quality training set.
2. **BERTweet fine‑tuning** yields a domain‑specific classifier.
3. **Batch inference** scores all 11 M tweets for injustice.
4. **Geolocation** extracts coordinates & FIPS codes, producing county‑level tallies.
5. **Validation & exploratory analysis** confirm robustness and surface insights.

---

## Repository Structure

```text
/README.md                  # ← You are here 📝
/.gitignore
/code
  ├── phase_1/               # Training‑data creation & unit tests
  │   ├── test_criteria.py
  │   ├── create_training_data.py
  │   └── test_training_pipeline.py
  ├── phase_2/               # Model fine‑tuning
  │   └── train_model.py
  ├── phase_3/               # Full‑dataset inference
  │   └── classify_all_tweets.py
  ├── phase_4/               # Geolocation & county aggregation
  │   └── add_location_data_fixed.py
  ├── phase_5/               # Analysis & sanity checks
  │   ├── analyze_existing_csv.py
  │   ├── analyze_tweet_dates.py
  │   ├── check_models.py
  │   └── explore_csv.py
  ├── phase_6/               # Alternative / legacy scripts (kept for posterity)
  │   ├── add_location_data.py
  │   ├── complete_location_and_county_analysis.py
  │   ├── complete_location_and_county_analysis_fixed.py
  │   └── complete_location_and_county_analysis_final.py
  └── phase_7/               # Supporting utilities
      ├── create_county_analysis.py
      ├── quick_county_analysis.py
      ├── modify_blm_criteria.py
      └── pilot_tweet.py
/data                        # Large CSVs (Git LFS or ignored)
  ├── training_data_20250623_151949.csv
  ├── all_tweets_classified_20250627_120357.csv
  ├── tweets_with_locations_fixed.csv
  └── blm_tweets_by_county.csv  ★ final deliverable
```

> **Note**: The `data/` directory is **not** stored in regular Git history to keep the repo small.  Either use **Git LFS** or download the files separately (see *Data* section).

---

## Setup & Dependencies

| Requirement                 | Version              |
| --------------------------- | -------------------- |
| Python                      | ≥ 3.9                |
| PyTorch                     | ≥ 2.0                |
| transformers                | ≥ 4.40               |
| pandas, numpy, scikit‑learn | latest               |
| geopy, us, shapely          | —                    |
| openai                      | ≥ 1.0 (Phase 1 only) |

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install core requirements
pip install -r requirements.txt  # file provided
```

If you plan to **re‑label tweets** you’ll also need an **OpenAI API key** exported as `OPENAI_API_KEY`.

---

## Data

1. **Raw tweets** (`tweets-*.jsonl`) are proprietary → not included.
2. **Pre‑labelled & processed CSVs** (see `/data` above) are >1 GB.  They can be downloaded from the project’s OSF page [link‑to‑be‑added] or recreated by running Phase 1–4.
3. County shapefiles come from the U.S. Census Bureau (freely available).

---

## Running the Pipeline

Below is the *happy‑path* workflow.  Each phase outputs artefacts consumed by the next stage.

### Phase 1 – Training‑Data Creation

*Create **`training_data_YYYYMMDD_HHMMSS.csv`*

```bash
cd code/phase_1
# (Optional) sanity‑check criteria
python test_criteria.py
# Generate labels with GPT‑4 (costs $$)
python create_training_data.py \
  --input ../../raw/tweets-0525-0615-*.jsonl \
  --out ../../data/training_data_$(date +%Y%m%d_%H%M%S).csv
```

### Phase 2 – Model Training

*Outputs **`bertweet_injustice_final/`*

```bash
cd ../phase_2
python train_model.py \
  --train_csv ../../data/training_data_20250623_151949.csv \
  --out_dir ../../models/bertweet_injustice_final
```

### Phase 3 – Full Classification

*Creates **`all_tweets_classified_YYYYMMDD_HHMMSS.csv`*

```bash
cd ../phase_3
python classify_all_tweets.py \
  --tweets ../../raw/*.jsonl \
  --model ../../models/bertweet_injustice_final \
  --out ../../data/all_tweets_classified_$(date +%Y%m%d_%H%M%S).csv
```

### Phase 4 – Location + County Mapping

*Outputs **`tweets_with_locations_fixed.csv`** & **`blm_tweets_by_county.csv`*

```bash
cd ../phase_4
python add_location_data_fixed.py \
  --classified ../../data/all_tweets_classified_20250627_120357.csv \
  --out_dir ../../data
```

### Phase 5 – Analysis & Validation (optional)

```bash
cd ../phase_5
python analyze_existing_csv.py --file ../../data/blm_tweets_by_county.csv
```

Feel free to run any script in isolation—each file has *docstrings* explaining arguments & expected outputs.

---

## Reproducibility

*Fix your random seeds* – each script sets `torch.manual_seed(42)` but feel free to override via CLI flags.\
If you deviate (e.g., different BERTweet checkpoint), update the experiment log in `experiments/`.

---

## Contact

| Name           | Email                                                    |
| -------------- | -------------------------------------------------------- |
| Your Name Here | [your.email@example.com](mailto\:your.email@example.com) |

Feedback, issues, and pull requests are welcome!

