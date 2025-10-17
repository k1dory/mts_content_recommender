"""
MTS Content Recommender - Training Script
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print(f"Random Seed: {RANDOM_SEED}")
print("="*80)

print("\n1. Loading data...")
interactions = pd.read_csv('interactions_train.csv')
users = pd.read_csv('users.csv')
items = pd.read_csv('items.csv')

print(f"Interactions: {interactions.shape}")
print(f"Users: {users.shape}")
print(f"Items: {items.shape}")

print("\n2. Merging data...")
data = interactions.merge(users, on='user_id', how='left')
data = data.merge(items, on='item_id', how='left')
print(f"Merged: {data.shape}")

print("\n3. Feature engineering...")

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

feature_cols = [
    'total_dur',
    'watch_year', 'watch_month', 'watch_day', 'watch_dayofweek', 'watch_hour',
    'age_encoded', 'income_encoded', 'sex_encoded', 'kids_flg',
    'content_type_encoded', 'release_year', 'age_rating', 'for_kids',
    'genres_count', 'countries_count', 'actors_count', 'keywords_count',
    'video_age', 'is_new_video'
]

X = data[feature_cols].fillna(-1)
y = data['watched_pct']

print(f"\nFeatures: {len(feature_cols)}")
print(f"Target: [{y.min():.1f}, {y.max():.1f}]")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

print(f"\nTrain: {X_train.shape}")
print(f"Val: {X_val.shape}")

print("\n4. Training LightGBM...")
model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    verbose=-1
)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)

y_pred_train = np.clip(y_pred_train, 0, 100)
y_pred_val = np.clip(y_pred_val, 0, 100)

y_train_binary = (y_train > 50).astype(int)
y_val_binary = (y_val > 50).astype(int)
y_pred_train_binary = (y_pred_train > 50).astype(int)
y_pred_val_binary = (y_pred_val > 50).astype(int)

train_acc = accuracy_score(y_train_binary, y_pred_train_binary)
val_acc = accuracy_score(y_val_binary, y_pred_val_binary)
val_auc = roc_auc_score(y_val_binary, y_pred_val)

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"Train Accuracy (>50%): {train_acc:.4f}")
print(f"Val Accuracy (>50%): {val_acc:.4f}")
print(f"Val AUC-ROC: {val_auc:.4f}")

print("\nTop 10 features:")
feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp.head(10))

print("\n5. Saving model...")
joblib.dump(model, f'model_seed_{RANDOM_SEED}.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')

print(f"\nSaved: model_seed_{RANDOM_SEED}.pkl, feature_cols.pkl")
print("="*80)
