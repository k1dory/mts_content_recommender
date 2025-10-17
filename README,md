 MTS Content Recommender - Video Watch Prediction

**Random Seed:** 42
**Task:** Predict if user will watch more than 50% of video
**Model:** LightGBM Regressor
**Val AUC-ROC:** 0.995

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train model

```bash
python train.py
```

Creates `model_seed_42.pkl` and `feature_cols.pkl`

### Generate predictions

```bash
python predict.py --model_path model_seed_42.pkl --seed 42
```

Creates `submission_seed_42.csv`

---

## Data

**Training data:**
- interactions_train.csv - 923k watch events
- users.csv - user demographics
- items.csv - video metadata

**Features:**
- user_id, item_id
- last_watch_dt - timestamp
- total_dur - video duration
- watched_pct - target (0-100%)

**User metadata:**
- age, income, sex, kids_flg

**Video metadata:**
- content_type, release_year, genres
- actors, directors, keywords, countries

---

## Feature Engineering

Total 20 features created:

**Temporal:** year, month, day, dayofweek, hour
**User:** age, income, sex, kids_flg (encoded)
**Video:** content_type, release_year, age_rating, for_kids (encoded)
**Counts:** genres, countries, actors, keywords
**Derived:** video_age, is_new_video

---

## Model

LightGBM with 200 trees, max_depth=8

Key parameters:
- learning_rate: 0.1
- subsample: 0.8
- random_state: 42

Top features by importance:
1. keywords_count
2. total_dur
3. release_year
4. actors_count
5. age_rating

---

## Results

Validation metrics:
- Accuracy (>50%): 96.75%
- AUC-ROC: 0.995

Submission stats:
- Mean watched: 42.3%
- Videos >50%: 39.7%

---

## Submission Format

CSV with columns: user_id, item_id, watched_pct

Values in range [0, 100]

---

## Requirements

- Test data not used for training
- Seed fixed at 42
- train.py runs end-to-end
- predict.py generates submission
- All results reproducible

---

**Hackathon:** MTS Web Services
**Date:** October 2025
