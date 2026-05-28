"""
OPENTIX 票價建議系統 — 模型訓練
輸入：features_program.csv, features_event.csv, opentix_detail.csv
輸出：model_summary.txt（訓練結果報告）

執行順序：
    1. opentix_scraper_v2.py
    2. opentix_detail_scraper_v3.py
    3. feature_engineering.py
    4. model_training.py   ← 本檔案

作者：Jin (TibaMe AI Data Scientist Program)
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════
# 載入資料
# ══════════════════════════════════════════════
print("=" * 55)
print("  OPENTIX 模型訓練")
print("=" * 55)

df_prog  = pd.read_csv("features_program.csv")
df_event = pd.read_csv("features_event.csv")

print(f"演出數：{len(df_prog)}")
print(f"場次數：{len(df_event)}")

# ══════════════════════════════════════════════
# PART A｜KMeans 市場區隔聚類
# ══════════════════════════════════════════════
print("\n── Part A：KMeans 聚類 ──")

le_cat   = LabelEncoder().fit(df_prog['category'])
le_scale = LabelEncoder().fit(df_prog['show_scale'])
le_tier  = LabelEncoder().fit(df_prog['venue_tier'])

df_prog['cat_enc']   = le_cat.transform(df_prog['category'])
df_prog['scale_enc'] = le_scale.transform(df_prog['show_scale'])
df_prog['tier_enc']  = le_tier.transform(df_prog['venue_tier'])

cluster_features = [
    'price_min', 'price_max', 'price_mean', 'price_spread',
    'zone_count_avg', 'num_events', 'num_cities',
    'advance_sale_days', 'discount_ratio',
    'cat_enc', 'scale_enc', 'tier_enc',
    'has_group', 'has_student', 'is_touring'
]

X_cluster = df_prog[cluster_features].fillna(0).values
scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(X_cluster)

# 找最佳 K
print("Silhouette scores:")
best_k, best_sil = 3, -1
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  k={k}  silhouette={sil:.3f}")
    if sil > best_sil:
        best_sil, best_k = sil, k

print(f"最佳 K = {best_k}（silhouette={best_sil:.3f}）")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_prog['cluster'] = km_final.fit_predict(X_scaled)

print("\n各群摘要：")
summary = df_prog.groupby('cluster').agg(
    count       = ('name',          'count'),
    top_cat     = ('category',      lambda x: x.value_counts().index[0]),
    top_scale   = ('show_scale',    lambda x: x.value_counts().index[0]),
    price_min_m = ('price_min',     'median'),
    price_max_m = ('price_max',     'median'),
    zone_avg    = ('zone_count_avg','median'),
    touring_pct = ('is_touring',    'mean'),
).round(1)
print(summary.to_string())

# ══════════════════════════════════════════════
# PART B｜相似度矩陣（供推薦使用）
# ══════════════════════════════════════════════
print("\n── Part B：相似度矩陣 ──")

sim_features = [
    'price_min', 'price_max', 'price_mean', 'zone_count_avg',
    'num_events', 'cat_enc', 'scale_enc', 'tier_enc'
]
X_sim = StandardScaler().fit_transform(df_prog[sim_features].fillna(0))
sim_matrix = cosine_similarity(X_sim)
print(f"相似度矩陣：{sim_matrix.shape}")

# ══════════════════════════════════════════════
# PART C｜售出率預測（RandomForest）
# ══════════════════════════════════════════════
print("\n── Part C：售出率預測模型 ──")

df_event['sellout_rate'] = (
    df_event['zones_sold_out'] / df_event['zone_count'].clip(lower=1)
)
df_event = df_event[df_event['zone_count'] > 0].copy()

le_cat_e   = LabelEncoder().fit(df_event['category'])
le_scale_e = LabelEncoder().fit(df_event['show_scale'])
le_tier_e  = LabelEncoder().fit(df_event['venue_tier'])
le_city_e  = LabelEncoder().fit(df_event['venue_city'].fillna('其他'))

df_event['cat_enc']   = le_cat_e.transform(df_event['category'])
df_event['scale_enc'] = le_scale_e.transform(df_event['show_scale'])
df_event['tier_enc']  = le_tier_e.transform(df_event['venue_tier'])
df_event['city_enc']  = le_city_e.transform(df_event['venue_city'].fillna('其他'))

pred_features = [
    'price_min', 'price_max', 'price_mean', 'price_spread', 'price_spread_ratio',
    'zone_count', 'has_multi_zone', 'has_premium', 'has_budget',
    'weekday', 'hour', 'month', 'is_weekend', 'is_matinee',
    'advance_sale_days', 'discount_ratio',
    'cat_enc', 'scale_enc', 'tier_enc', 'city_enc',
]

X = df_event[pred_features].fillna(0)
y = df_event['sellout_rate']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_absolute_error')
mae = -cv_scores.mean()
print(f"RandomForest 5-fold MAE：{mae:.3f} ± {cv_scores.std():.3f}")

rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=pred_features)
print("\n特徵重要性（前10）：")
print(importances.sort_values(ascending=False).head(10).round(3).to_string())

# 各類別 × 規模基準售出率
baseline = df_event.groupby(['category','show_scale'])['sellout_rate'].agg(
    ['mean','median','std','count']
).round(3)
print("\n各類別 × 規模基準售出率：")
print(baseline.to_string())

# ══════════════════════════════════════════════
# 輸出訓練報告
# ══════════════════════════════════════════════
df_prog.to_csv("features_clustered.csv", index=False, encoding="utf-8-sig")

report = f"""OPENTIX 模型訓練報告
{'='*50}
資料規模
  演出數：{len(df_prog)}
  場次數：{len(df_event)}

Part A：KMeans 聚類
  最佳 K：{best_k}
  Silhouette Score：{best_sil:.3f}
  各群分布：
{summary[['count','top_cat','price_min_m','price_max_m']].to_string()}

Part B：相似度矩陣
  維度：{sim_matrix.shape}

Part C：售出率預測
  模型：RandomForest（100棵樹）
  5-fold MAE：{mae:.3f}
  最重要特徵：{importances.sort_values(ascending=False).index[0]}（{importances.sort_values(ascending=False).iloc[0]:.3f}）

注意事項
  售出率目標變數基於爬取當下的 zone_remaining，
  非演出結束後的真實售票結果，屬代理指標（proxy metric）。
  請搭配 OPENTIX 年度報告市場基準數字解讀。
"""

with open("model_summary.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("\n✅ 訓練完成")
print("   輸出：features_clustered.csv, model_summary.txt")
