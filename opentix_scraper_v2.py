"""
OPENTIX 演出資料爬蟲 v2（基於真實 API）
API: https://csm.api.opentix.life/programs?page=1&rowCount=10

作者：Jin (TibaMe AI Data Scientist Program)
"""

import requests
import pandas as pd
import time
from datetime import datetime

# ── 設定 ──────────────────────────────────────────────
API_BASE = "https://csm.api.opentix.life/programs"
ROW_COUNT = 50          # 每頁幾筆（可試試 50 或 100）
DELAY = 1.2             # 每次請求間隔秒數
OUTPUT_FILE = "opentix_data.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.opentix.life/",
    "Origin": "https://www.opentix.life",
    "Accept": "application/json",
}

# 目標類別（依回傳資料中 displayCategory 的實際值）
TARGET_CATEGORIES = ["戲劇", "音樂劇", "舞蹈", "傳統戲曲", "親子"]


# ── Unix timestamp 轉換 ────────────────────────────────
def ts_to_dt(ts):
    """Unix timestamp → 可讀日期時間字串"""
    if not ts:
        return None
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


# ── 單頁抓取 ───────────────────────────────────────────
def fetch_page(page: int, category: str = None) -> dict | None:
    """
    呼叫 API 取得一頁資料
    category: 若有，加入 displayCategory 篩選（需確認 API 是否支援）
    """
    params = {
        "page": page,
        "rowCount": ROW_COUNT,
    }
    # 嘗試加入類別篩選（依 API 是否支援調整參數名稱）
    if category:
        params["displayCategory"] = category  # 或 "category"、"type" 等，需測試

    try:
        resp = requests.get(API_BASE, headers=HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"    [HTTP Error] {e}")
        return None
    except Exception as e:
        print(f"    [Error] {e}")
        return None


# ── 解析單筆演出 ───────────────────────────────────────
def parse_program(item: dict) -> dict:
    """將 API 回傳的一筆演出資料轉為 DataFrame row"""
    events = item.get("events") or []

    return {
        "program_id":       item.get("id", ""),
        "name":             item.get("name", ""),
        "category":         item.get("displayCategory", ""),
        "cities":           ", ".join(item.get("cities") or []),
        "max_price":        item.get("maxPrice"),
        "min_price":        item.get("minPrice"),
        "film_rating":      item.get("filmRating", ""),
        "age_restriction":  item.get("ageRestriction", ""),
        "start_datetime":   ts_to_dt(item.get("startDateTime")),
        "end_datetime":     ts_to_dt(item.get("endDateTime")),
        "num_performances": len(events),   # 場次數量
        "image_url":        item.get("imageUrl", ""),
        "scraped_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── 翻頁抓取所有資料（不篩類別）─────────────────────────
def fetch_all_programs() -> list[dict]:
    """
    不帶類別篩選，抓取全部演出，再用 pandas 過濾
    這樣最穩定，避免 API 不支援 category 參數的問題
    """
    all_items = []
    page = 1

    print("🔄 開始抓取全部演出資料...")

    while True:
        print(f"  第 {page} 頁...", end=" ", flush=True)
        data = fetch_page(page)

        if not data or data.get("error"):
            print(f"錯誤：{data}")
            break

        result = data.get("result", {})
        items = result.get("data", [])
        next_page = result.get("nextPage")

        if not items:
            print("無資料，停止")
            break

        all_items.extend(items)
        print(f"取得 {len(items)} 筆（累計 {len(all_items)} 筆）")

        # 沒有下一頁了
        if not next_page:
            print("  已是最後一頁")
            break

        page = next_page
        time.sleep(DELAY)

    return all_items


# ── 主流程 ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  OPENTIX 爬蟲 v2 啟動")
    print("  API: csm.api.opentix.life/programs")
    print("=" * 60)

    # 1. 抓取所有資料
    raw_items = fetch_all_programs()

    if not raw_items:
        print("\n⚠️  未取得任何資料")
        return

    # 2. 解析為 DataFrame
    rows = [parse_program(item) for item in raw_items]
    df = pd.DataFrame(rows)

    print(f"\n✅ 總共取得 {len(df)} 筆演出")
    print(f"   類別分布：\n{df['category'].value_counts().to_string()}")

    # 3. 儲存完整版
    df.to_csv("opentix_all.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 完整資料已存：opentix_all.csv")

    # 4. 篩選舞台劇相關類別
    df_theater = df[df["category"].isin(TARGET_CATEGORIES)].copy()
    df_theater.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"💾 舞台劇資料已存：{OUTPUT_FILE}（{len(df_theater)} 筆）")

    # 5. 基本統計預覽
    print(f"\n📊 舞台劇資料統計：")
    print(f"   最低票價範圍：{df_theater['min_price'].min()} ～ {df_theater['min_price'].max()} 元")
    print(f"   最高票價範圍：{df_theater['max_price'].min()} ～ {df_theater['max_price'].max()} 元")
    print(f"   城市分布：{df_theater['cities'].value_counts().head(5).to_dict()}")
    print(f"\n📋 資料預覽：")
    print(df_theater[["name", "category", "cities", "min_price", "max_price",
                       "num_performances", "start_datetime"]].head(10).to_string())


if __name__ == "__main__":
    run()
