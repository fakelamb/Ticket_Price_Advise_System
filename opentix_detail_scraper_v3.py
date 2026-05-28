"""
OPENTIX 演出詳細資料爬蟲 v3
API: https://csm.api.opentix.life/programs/{id}

從 opentix_all.csv 讀取所有演出 ID，
逐一抓取各票區、場地、剩餘票數等詳細資訊

作者：Jin (TibaMe AI Data Scientist Program)
"""

import requests
import pandas as pd
import time
from datetime import datetime

# ── 設定 ──────────────────────────────────────────────
INPUT_FILE  = "opentix_all.csv"         # 第一支爬蟲產生的檔案
OUTPUT_FILE = "opentix_detail.csv"      # 本支爬蟲輸出
TARGET_CATEGORIES = ["戲劇", "音樂劇", "舞蹈", "傳統戲曲", "親子"]
DELAY = 1.5                             # 每次請求間隔秒數

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.opentix.life/",
    "Accept": "application/json",
}

def ts(unix):
    """Unix timestamp → 日期時間字串"""
    try:
        return datetime.fromtimestamp(unix).strftime("%Y-%m-%d %H:%M") if unix else None
    except:
        return None


# ── 抓取單筆演出詳細資料 ───────────────────────────────
def fetch_detail(program_id: str) -> dict | None:
    url = f"https://csm.api.opentix.life/programs/{program_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result")
    except Exception as e:
        print(f"    [Error] {program_id}: {e}")
        return None


# ── 解析詳細資料 → 每票區一筆 row ─────────────────────
def parse_detail(program_id: str, name: str, category: str, detail: dict) -> list[dict]:
    rows = []

    # 基本資訊
    base = {
        "program_id":        program_id,
        "name":              name,
        "category":          category,
        "sale_start":        ts(detail.get("onlineStartTime")),
        "sale_end":          ts(detail.get("onlineEndTime")),
        "age_restriction":   detail.get("ageRestriction", ""),
        "film_rating":       detail.get("filmRating", ""),
        "has_group_discount": False,
        "has_student_discount": False,
        "has_disability_discount": False,
    }

    # 逐個場館（eventVenues）
    for ev in detail.get("eventVenues") or []:
        venue = ev.get("venue") or {}
        venue_name = venue.get("name", "")
        venue_city = venue.get("city", "")
        venue_area = venue.get("area", "")
        venue_addr = venue.get("address", "")
        venue_lat  = venue.get("lat")
        venue_lng  = venue.get("lng")

        # 逐個場次（events）
        for event in ev.get("events") or []:
            event_id  = event.get("id", "")
            event_start = ts(event.get("startDateTime"))
            event_end   = ts(event.get("endDateTime"))
            total_remaining = (event.get("quantity") or {}).get("remainingQuantity")
            is_unlimited    = (event.get("quantity") or {}).get("unlimited", False)

            # 逐個票區（groupSections.default）
            sections = (event.get("groupSections") or {}).get("default") or []
            for section in sections:
                # 跳過輪椅席（type != 0）
                section_type = section.get("type", 0)
                if section_type not in [0]:  # 0=一般, 1=輪椅, 2=輪椅陪同
                    continue

                zone_name       = section.get("name", "")
                zone_price      = section.get("price")
                zone_remaining  = (section.get("quantity") or {}).get("remainingQuantity")
                zone_unlimited  = (section.get("quantity") or {}).get("unlimited", False)

                # 分析優惠方案
                price_plans = section.get("pricePlans") or []
                plan_names  = [p.get("name", "") for p in price_plans]
                discounts   = [p.get("currentDiscount") for p in price_plans if p.get("currentDiscount")]
                best_prices = [p.get("bestPrice") for p in price_plans if p.get("bestPrice")]

                has_group    = any("團體" in n for n in plan_names)
                has_student  = any("青年" in n or "學生" in n for n in plan_names)
                has_disabled = any("身障" in n for n in plan_names)

                row = {
                    **base,
                    # 場地
                    "venue_name": venue_name,
                    "venue_city": venue_city,
                    "venue_area": venue_area,
                    "venue_address": venue_addr,
                    "venue_lat": venue_lat,
                    "venue_lng": venue_lng,
                    # 場次
                    "event_id": event_id,
                    "event_start": event_start,
                    "event_end": event_end,
                    "event_total_remaining": total_remaining if not is_unlimited else -1,
                    # 票區
                    "zone_name": zone_name,
                    "zone_price": zone_price,
                    "zone_remaining": zone_remaining if not zone_unlimited else -1,
                    "zone_is_sold_out": (zone_remaining == 0) if not zone_unlimited else False,
                    # 優惠
                    "has_group_discount": has_group,
                    "has_student_discount": has_student,
                    "has_disability_discount": has_disabled,
                    "min_discount_price": min(best_prices) if best_prices else None,
                    "num_price_plans": len(price_plans),
                    # 爬取時間
                    "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                rows.append(row)

    return rows


# ── 主流程 ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  OPENTIX 詳細票區爬蟲 v3 啟動")
    print("=" * 60)

    # 讀取演出清單
    df_all = pd.read_csv(INPUT_FILE)
    df_theater = df_all[df_all["category"].isin(TARGET_CATEGORIES)].copy()
    total = len(df_theater)
    print(f"目標演出數：{total} 筆（{', '.join(TARGET_CATEGORIES)}）\n")

    all_rows = []
    failed   = []

    for i, row in enumerate(df_theater.itertuples(), 1):
        pid   = str(row.program_id)
        name  = row.name
        cat   = row.category

        print(f"[{i:3d}/{total}] {name[:30]}...", end=" ", flush=True)
        detail = fetch_detail(pid)

        if not detail:
            print("❌ 失敗")
            failed.append(pid)
            time.sleep(DELAY)
            continue

        rows = parse_detail(pid, name, cat, detail)
        all_rows.extend(rows)
        print(f"✅ {len(rows)} 筆票區資料")
        time.sleep(DELAY)

    # ── 儲存結果 ──
    if not all_rows:
        print("\n⚠️  沒有取得任何資料")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print(f"  完成！")
    print(f"  總票區筆數：{len(df)}")
    print(f"  涵蓋演出數：{df['program_id'].nunique()}")
    print(f"  涵蓋場館數：{df['venue_name'].nunique()}")
    print(f"  失敗演出數：{len(failed)}")
    if failed:
        print(f"  失敗 ID：{failed[:5]}...")
    print(f"  輸出檔案：{OUTPUT_FILE}")
    print("=" * 60)

    # 快速統計
    print(f"\n📊 票區統計：")
    print(f"  已售完票區比例：{df['zone_is_sold_out'].mean():.1%}")
    print(f"  有團體優惠的演出：{df.groupby('program_id')['has_group_discount'].any().sum()} 個")
    print(f"  有青年/學生優惠的演出：{df.groupby('program_id')['has_student_discount'].any().sum()} 個")
    print(f"\n  各票區平均票價：")
    print(df.groupby("zone_name")["zone_price"].agg(["mean","count"]).sort_values("count", ascending=False).head(10).round(0))


if __name__ == "__main__":
    run()
