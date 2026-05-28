"""
OPENTIX 票價建議系統 — 特徵工程
輸入：opentix_detail.csv
輸出：features_program.csv（演出層級）
      features_event.csv（場次層級）

作者：Jin (TibaMe AI Data Scientist Program)
"""

import pandas as pd
import numpy as np

# ── 載入資料 ───────────────────────────────────────────
df = pd.read_csv("opentix_detail.csv")
df['event_start'] = pd.to_datetime(df['event_start'])
df['event_end']   = pd.to_datetime(df['event_end'])
df['sale_start']  = pd.to_datetime(df['sale_start'])

print(f"原始資料：{len(df)} 筆，{df['program_id'].nunique()} 個演出，{df['event_id'].nunique()} 場次")

# ══════════════════════════════════════════════════════
# STEP 1｜市場基準票價表（來自 OPENTIX 2024 年度報告）
# ══════════════════════════════════════════════════════
market_benchmark = pd.DataFrame([
    # 大型節目（可售票數 800 張以上）
    {'category': '戲劇', 'scale': '大型', 'market_median_price': 1008},
    {'category': '音樂', 'scale': '大型', 'market_median_price': 683},
    {'category': '舞蹈', 'scale': '大型', 'market_median_price': 822},
    {'category': '親子', 'scale': '大型', 'market_median_price': 716},
    # 中型節目（200～799 張）
    {'category': '戲劇', 'scale': '中型', 'market_median_price': 741},
    {'category': '音樂', 'scale': '中型', 'market_median_price': 473},
    {'category': '舞蹈', 'scale': '中型', 'market_median_price': 476},
    {'category': '親子', 'scale': '中型', 'market_median_price': 490},
    # 小型節目（199 張以下）
    {'category': '戲劇', 'scale': '小型', 'market_median_price': 530},
    {'category': '音樂', 'scale': '中型', 'market_median_price': 496},
    {'category': '舞蹈', 'scale': '小型', 'market_median_price': 531},
    {'category': '親子', 'scale': '小型', 'market_median_price': 426},
])

# ══════════════════════════════════════════════════════
# STEP 2｜場館分級
# ══════════════════════════════════════════════════════
# 國家級場館：兩廳院、臺中歌劇院、衛武營、北藝中心
national_venues = [
    '國家戲劇院', '國家音樂廳', '國家兩廳院實驗劇場', '演奏廳',
    '臺中國家歌劇院大劇院', '臺中國家歌劇院中劇院', '臺中國家歌劇院小劇場',
    '衛武營國家藝術文化中心戲劇院', '衛武營國家藝術文化中心歌劇院',
    '衛武營國家藝術文化中心音樂廳', '衛武營國家藝術文化中心表演廳',
    '臺北表演藝術中心 大劇院', '臺北表演藝術中心 球劇場', '臺北表演藝術中心 中劇院',
]
# 市立/縣立場館
municipal_venues = [
    '臺北市藝文推廣處城市舞台', '臺北市政大樓親子劇場', '大稻埕戲苑',
    '臺灣戲曲中心大表演廳', '臺灣戲曲中心多功能廳',
    '臺中市中山堂', '臺南文化中心演藝廳',
]

def classify_venue(name):
    if any(v in name for v in national_venues):
        return '國家級'
    elif any(v in name for v in municipal_venues):
        return '市立級'
    else:
        return '民間場館'

df['venue_tier'] = df['venue_name'].apply(classify_venue)

# ══════════════════════════════════════════════════════
# STEP 3｜場次層級特徵
# ══════════════════════════════════════════════════════
event_features = df.groupby('event_id').agg(
    program_id        = ('program_id', 'first'),
    name              = ('name', 'first'),
    category          = ('category', 'first'),
    venue_name        = ('venue_name', 'first'),
    venue_city        = ('venue_city', 'first'),
    venue_tier        = ('venue_tier', 'first'),
    event_start       = ('event_start', 'first'),
    event_end         = ('event_end', 'first'),
    sale_start        = ('sale_start', 'first'),
    has_group         = ('has_group_discount', 'first'),
    has_student       = ('has_student_discount', 'first'),
    has_disability    = ('has_disability_discount', 'first'),
    # 票價統計
    zone_count        = ('zone_name', 'count'),
    price_min         = ('zone_price', 'min'),
    price_max         = ('zone_price', 'max'),
    price_mean        = ('zone_price', 'mean'),
    price_median      = ('zone_price', 'median'),
    price_std         = ('zone_price', 'std'),
    # 剩餘票數
    total_remaining   = ('zone_remaining', 'sum'),
    zones_sold_out    = ('zone_is_sold_out', 'sum'),
).reset_index()

# ── 衍生時間特徵 ───────────────────────────────────────
event_features['weekday']       = event_features['event_start'].dt.dayofweek   # 0=週一
event_features['weekday_name']  = event_features['event_start'].dt.day_name()
event_features['hour']          = event_features['event_start'].dt.hour
event_features['month']         = event_features['event_start'].dt.month
event_features['is_weekend']    = event_features['weekday'].isin([5, 6]).astype(int)
event_features['is_matinee']    = (event_features['hour'] < 15).astype(int)    # 下午3點前為日場

# 季節
def get_season(month):
    if month in [3, 4, 5]:   return '春'
    elif month in [6, 7, 8]: return '夏'
    elif month in [9,10,11]: return '秋'
    else:                     return '冬'
event_features['season'] = event_features['month'].apply(get_season)

# 距今天的天數（目前資料都是未來場次）
today = pd.Timestamp.now()
event_features['days_until_show'] = (event_features['event_start'] - today).dt.days

# 售票提前天數
event_features['advance_sale_days'] = (
    event_features['event_start'] - event_features['sale_start']
).dt.days.clip(lower=0)

# ── 衍生票價特徵 ───────────────────────────────────────
event_features['price_spread']      = event_features['price_max'] - event_features['price_min']
event_features['price_spread_ratio'] = (
    event_features['price_spread'] / event_features['price_mean'].replace(0, np.nan)
)
event_features['has_multi_zone']    = (event_features['zone_count'] > 1).astype(int)
event_features['has_premium']       = (event_features['price_max'] >= 2000).astype(int)
event_features['has_budget']        = (event_features['price_min'] <= 500).astype(int)
event_features['discount_ratio']    = (
    event_features['has_student'].astype(int) +
    event_features['has_group'].astype(int) +
    event_features['has_disability'].astype(int)
)

# ── 場館規模（用票區總剩餘量估計）─────────────────────
# 用 zone_count × 平均每區剩餘 推估可售票數
event_features['est_capacity'] = event_features['total_remaining']  # 保守估計（還沒賣的）

def classify_scale(row):
    # 以票價差距和票區數量估計規模
    if row['zone_count'] >= 6 or row['price_max'] >= 3000:
        return '大型'
    elif row['zone_count'] >= 3 or row['price_max'] >= 1200:
        return '中型'
    else:
        return '小型'

event_features['show_scale'] = event_features.apply(classify_scale, axis=1)

# ══════════════════════════════════════════════════════
# STEP 4｜演出層級特徵
# ══════════════════════════════════════════════════════
program_features = event_features.groupby('program_id').agg(
    name              = ('name', 'first'),
    category          = ('category', 'first'),
    venue_name        = ('venue_name', 'first'),
    venue_city        = ('venue_city', 'first'),
    venue_tier        = ('venue_tier', 'first'),
    show_scale        = ('show_scale', 'first'),
    num_events        = ('event_id', 'count'),           # 總場次數
    num_cities        = ('venue_city', 'nunique'),       # 幾個城市巡演
    price_min         = ('price_min', 'min'),
    price_max         = ('price_max', 'max'),
    price_mean        = ('price_mean', 'mean'),
    price_spread      = ('price_spread', 'max'),
    zone_count_avg    = ('zone_count', 'mean'),
    has_weekend       = ('is_weekend', 'max'),
    has_matinee       = ('is_matinee', 'max'),
    has_group         = ('has_group', 'first'),
    has_student       = ('has_student', 'first'),
    has_disability    = ('has_disability', 'first'),
    discount_ratio    = ('discount_ratio', 'first'),
    advance_sale_days = ('advance_sale_days', 'mean'),
    total_remaining   = ('total_remaining', 'sum'),
    zones_sold_out    = ('zones_sold_out', 'sum'),
).reset_index()

# 是否跨城市巡演
program_features['is_touring'] = (program_features['num_cities'] > 1).astype(int)

# 場次規模分類（用場次數）
def classify_num_events(n):
    if n == 1:   return '單場'
    elif n <= 3: return '小規模'
    elif n <= 8: return '中規模'
    else:        return '大規模'
program_features['event_scale'] = program_features['num_events'].apply(classify_num_events)

# ══════════════════════════════════════════════════════
# STEP 5｜merge 市場基準票價
# ══════════════════════════════════════════════════════
program_features = program_features.merge(
    market_benchmark,
    on=['category', 'scale'] if 'scale' in market_benchmark.columns
    else None,
    how='left'
) if False else program_features  # placeholder

# 手動 merge（依 category + show_scale）
market_benchmark = market_benchmark.rename(columns={'scale': 'show_scale'})
program_features = program_features.merge(
    market_benchmark, on=['category', 'show_scale'], how='left'
)

# 票價 vs 市場基準的比值
program_features['price_vs_market'] = (
    program_features['price_mean'] / program_features['market_median_price']
).round(3)

# 票價高於市場（over-priced / under-priced / aligned）
def price_position(ratio):
    if pd.isna(ratio):   return '未知'
    elif ratio > 1.2:    return '高於市場'
    elif ratio < 0.8:    return '低於市場'
    else:                return '市場行情'
program_features['price_position'] = program_features['price_vs_market'].apply(price_position)

# ══════════════════════════════════════════════════════
# STEP 6｜輸出
# ══════════════════════════════════════════════════════
event_features.to_csv("features_event.csv", index=False, encoding="utf-8-sig")
program_features.to_csv("features_program.csv", index=False, encoding="utf-8-sig")

print("\n✅ 特徵工程完成！")
print(f"\n📋 場次層級特徵（features_event.csv）")
print(f"   筆數：{len(event_features)}")
print(f"   欄位：{len(event_features.columns)}")
print(f"   欄位清單：{list(event_features.columns)}")

print(f"\n📋 演出層級特徵（features_program.csv）")
print(f"   筆數：{len(program_features)}")
print(f"   欄位：{len(program_features.columns)}")
print(f"   欄位清單：{list(program_features.columns)}")

print(f"\n📊 演出規模分布：")
print(program_features['show_scale'].value_counts())
print(f"\n📊 票價位置分布：")
print(program_features['price_position'].value_counts())
print(f"\n📊 場館等級分布：")
print(program_features['venue_tier'].value_counts())
print(f"\n📊 各類別 × 規模平均票價 vs 市場基準：")
summary = program_features.groupby(['category','show_scale'])[['price_mean','market_median_price','price_vs_market']].mean().round(0)
print(summary)
