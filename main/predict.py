"""
MTS Content Recommender - Prediction Script
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path

RANDOM_SEED = 42

def preprocess_test_data(interactions, users, items, feature_cols):
    data = interactions.merge(users, on='user_id', how='left')
    data = data.merge(items, on='item_id', how='left')

    data['last_watch_dt'] = pd.to_datetime(data['last_watch_dt'])
    data['watch_year'] = data['last_watch_dt'].dt.year
    data['watch_month'] = data['last_watch_dt'].dt.month
    data['watch_day'] = data['last_watch_dt'].dt.day
    data['watch_dayofweek'] = data['last_watch_dt'].dt.dayofweek
    data['watch_hour'] = data['last_watch_dt'].dt.hour

    data['age_encoded'] = data['age'].astype('category').cat.codes
    data['income_encoded'] = data['income'].astype('category').cat.codes
    data['sex_encoded'] = data['sex'].map({'М': 0, 'Ж': 1}).fillna(-1)

    data['content_type_encoded'] = data['content_type'].astype('category').cat.codes
    data['release_year'] = data['release_year'].fillna(2000)
    data['age_rating'] = data['age_rating'].fillna(0)
    data['for_kids'] = data['for_kids'].fillna(0).astype(int)

    data['genres_count'] = data['genres'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    data['countries_count'] = data['countries'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    data['actors_count'] = data['actors'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    data['keywords_count'] = data['keywords'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

    data['video_age'] = data['watch_year'] - data['release_year']
    data['is_new_video'] = (data['video_age'] <= 1).astype(int)

    X = data[feature_cols].fillna(-1)

    return X, data[['user_id', 'item_id']]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='interactions_public_test.csv')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = f'submission_seed_{args.seed}.csv'

    print("="*80)
    print("MTS CONTENT RECOMMENDER - PREDICTION")
    print("="*80)

    if not Path(args.test_path).exists():
        print(f"ERROR: {args.test_path} not found")
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"ERROR: {args.model_path} not found")
        sys.exit(1)

    print(f"\nLoading model: {args.model_path}")
    model = joblib.load(args.model_path)
    feature_cols = joblib.load('feature_cols.pkl')
    print(f"Features: {len(feature_cols)}")

    print(f"\nLoading test data: {args.test_path}")
    interactions = pd.read_csv(args.test_path)
    users = pd.read_csv('users.csv')
    items = pd.read_csv('items.csv')

    print(f"Test: {interactions.shape}")

    print("\nPreprocessing...")
    X_test, ids = preprocess_test_data(interactions, users, items, feature_cols)

    print("\nPredicting...")
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, 100)

    submission = pd.DataFrame({
        'user_id': ids['user_id'],
        'item_id': ids['item_id'],
        'watched_pct': predictions
    })

    submission.to_csv(args.output_path, index=False)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Saved: {args.output_path}")
    print(f"Predictions: {len(submission)}")
    print(f"\nStats:")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Median: {np.median(predictions):.2f}")
    print(f"  >50%: {(predictions > 50).sum()} ({(predictions > 50).mean()*100:.1f}%)")
    print("\nFirst 5:")
    print(submission.head())
    print("="*80)


if __name__ == "__main__":
    main()
