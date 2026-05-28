# 🎭 Taiwan Stage Performance Ticket Price Advisor

> A data science portfolio project analyzing Taiwan's performing arts ticketing market (OPENTIX platform) to help ticketing administrators design optimal pricing structures and forecast sellout rates.

---

## 📌 Project Overview

Taiwan's performing arts market hit a record-high box office of NT$1.84 billion in 2024, with over 3,150 productions and 2.1 million tickets sold (OPENTIX Annual Report, 2024). Yet many small and mid-size productions struggle with pricing strategy — setting prices too high risks low attendance, while setting them too low leaves revenue on the table.

This project builds a **ticket price advisory and sellout rate prediction system** based on real market data scraped from OPENTIX (`opentix.life`), Taiwan's leading performing arts ticketing platform.

**Target user:** Ticketing administrators planning a new production who want data-backed guidance on pricing zones and sellout potential.

---

## 🗂️ Project Structure

```
opentix-price-advisor/
│
├── README.md
│
├── scraper/
│   ├── opentix_scraper_v2.py          # Scrapes production listing (REST API)
│   └── opentix_detail_scraper_v3.py   # Scrapes per-zone ticket details
│
├── analysis/
│   └── feature_engineering.py         # Feature construction + market benchmark merge
│
├── app/
│   └── opentix_predictor.html         # Interactive prediction + AI suggestion UI
│
└── data/
    └── opentix_all.csv                # Raw scraped data (319 productions)
```

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| Source | OPENTIX (`csm.api.opentix.life`) |
| Productions | 319 |
| Performances | 1,209 |
| Venues | 120 |
| Ticket zone records | 4,554 |
| Categories | 戲劇 (Drama), 舞蹈 (Dance), 親子 (Family) |
| Scraped | April 2026 |

### Key Fields Collected

| Field | Description |
|-------|-------------|
| `zone_name` | Ticket zone label (e.g., A區, VIP席) |
| `zone_price` | Face value (NTD) |
| `zone_remaining` | Remaining tickets at time of scrape |
| `venue_name` / `venue_city` | Venue and city |
| `event_start` | Performance datetime |
| `has_student_discount` | Youth/student pricing available |
| `has_group_discount` | Group pricing available |
| `sale_start` | Ticket sale start date |

---

## ⚙️ Methodology

### 1. Data Collection

OPENTIX does not publish historical sales data. This project uses two REST API endpoints discovered via Chrome DevTools:

```
GET https://csm.api.opentix.life/programs?page=1&rowCount=50
GET https://csm.api.opentix.life/programs/{id}
```

Productions are scraped across performing arts categories, then detailed ticket zone data is fetched per production. A polite delay of 1.5 seconds between requests is applied throughout.

### 2. Feature Engineering

Features are constructed at two levels:

**Event level (per performance):**
- Time features: weekday, hour, month, season, is_weekend, is_matinee
- Pricing features: min/max/mean/spread, zone count, has_premium, has_budget
- Venue tier: National (兩廳院, 衛武營, 北藝中心), Municipal, Independent
- Advance sale days (strongest predictor per model, importance: 25.9%)

**Production level (per show):**
- Number of performances, touring cities, discount types
- Market position: price vs. OPENTIX 2024 market median benchmark

### 3. Market Benchmark Integration

OPENTIX's 2024 Annual Report (commissioned by NYCU Statistics Institute) provides median ticket prices by category and venue size. These are merged as external benchmark features:

| Category | Large (800+ seats) | Medium (200–799) | Small (≤199) |
|----------|-------------------|-----------------|--------------|
| Drama    | NT$1,008          | NT$741          | NT$530       |
| Dance    | NT$822            | NT$476          | NT$531       |
| Family   | NT$716            | NT$490          | NT$426       |

### 4. Sellout Rate Prediction

A **Random Forest Regressor** (MAE = 0.203) is trained on event-level features to estimate sellout rate (proportion of ticket zones sold out at the time of scrape).

**⚠️ Important Limitation:** Because OPENTIX does not retain historical sales snapshots, `zone_remaining` reflects the *current* state at scrape time — not the final state after a production closes. The sellout rate used here is a **proxy metric**, not a ground-truth outcome. This is transparently disclosed in the application UI and accounted for via market benchmark calibration from the 2024 annual report (market average: 24% for large drama, 24% for medium, 15% for small productions).

Top predictive features:
1. Advance sale days (25.9%)
2. Month (11.3%)
3. Max ticket price (8.8%)
4. Zone count (7.9%)
5. Mean price (6.3%)

### 5. Recommendation System

Similar productions are retrieved using **cosine similarity** on scaled feature vectors, filtered by category and scale. Median prices from the top-N similar productions form the zone price recommendations.

### 6. AI Advisory Layer

User inputs and prediction outputs are passed to the Claude API (claude-sonnet) via streaming, which generates specific, labeled suggestions (⚠️ Warning / 💡 Suggestion / ✅ Strength) based on the pricing structure and market context.

---

## 🖥️ Application Features

The interactive HTML application (`app/opentix_predictor.html`) allows users to:

- **Input their own ticket zone names and prices** (add/remove zones freely)
- **Set performance conditions**: date, time, venue tier, city, advance sale window, discounts
- **Receive a sellout rate prediction** with animated gauge and risk classification
- **See factor-by-factor breakdown**: which conditions help or hurt predicted sellout rate
- **Compare each zone price against the 2024 market median**
- **Get AI-generated suggestions** streamed in real-time via Claude API

---

## 🚀 How to Run

### Scraper

```bash
pip install requests pandas
python scraper/opentix_scraper_v2.py        # Step 1: production listings
python scraper/opentix_detail_scraper_v3.py # Step 2: zone details
```

Output: `opentix_all.csv`, `opentix_detail.csv`

### Feature Engineering

```bash
python analysis/feature_engineering.py
```

Output: `features_event.csv`, `features_program.csv`

### Application

Open `app/opentix_predictor.html` in any modern browser. No server required — all logic runs client-side.

> The AI suggestion feature calls the Anthropic API and requires an API key injected at runtime or via a backend proxy.

---

## 📈 Key Findings

- **Advance sale window is the strongest predictor** of sellout rate (feature importance: 25.9%). Productions selling tickets 90+ days out show ~6 percentage points higher sellout rates.
- **Multi-zone pricing correlates positively with sellout rate** (r = +0.193). Single-price productions show lower rates, possibly due to narrower audience reach.
- **High-price ceiling signals quality**: productions with a max price ≥ NT$2,000 show higher sellout rates, reflecting brand premium effects.
- **Taipei has lower sellout rates than Taichung** (18% vs. 32%), likely due to higher supply concentration in Taipei.
- **Youth discounts are associated with ~4 percentage points higher sellout rates** vs. productions without.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Scraping, feature engineering, modeling |
| requests / pandas | Data collection and processing |
| scikit-learn | RandomForest, KMeans clustering, cosine similarity |
| HTML / CSS / Vanilla JS | Interactive application |
| Claude API (claude-sonnet) | AI-generated pricing suggestions |

---

## ⚠️ Data & Ethical Notes

- All data scraped from publicly accessible OPENTIX endpoints with polite rate limiting (1.5s delay).
- No user data, purchase records, or private information was collected.
- Predictions are market estimates for planning purposes only and should not be treated as guarantees.
- OPENTIX 2024 Annual Report data cited with source attribution.

---

## 👤 Author

**Jin**
TibaMe AI Data Scientist Professional Program
Background: Theater lighting design & cultural venue management → Data Science

This project applies domain expertise in Taiwan's performing arts industry to a data science context, bridging practical ticketing knowledge with machine learning methodology.

---

## 📄 License

MIT License — feel free to fork, adapt, and build on this work.
