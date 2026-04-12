# Equintel

Horse racing prediction system using XGBoost. Predicts win/place probability for each runner and identifies value bets by comparing model fair odds to bookmaker odds.

## How it works

1. **Historical data** — Kaggle SQLite dataset (~466 MB, UK/Ireland 2015-2026)
2. **Feature engineering** — Form ratings, jockey/trainer stats, going preference, market implied probability, draw position, days since last run, and more
3. **Model** — Two calibrated XGBoost classifiers: one for win, one for place
4. **Live inference** — The Racing API supplies today's racecards
5. **Value detection** — Edge = (model probability × bookie odds) − 1; flag where edge > threshold

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the historical data

Download from Kaggle (free account required):
[Horse Racing results - UK/Ireland 2015-2026](https://www.kaggle.com/datasets/deltaromeo/horse-racing-results-ukireland-2015-2025)

Extract `raceform.db` into the `data/` directory:

```
data/raceform.db
```

### 3. Set up The Racing API

Sign up for a free account at [theracingapi.com](https://www.theracingapi.com), then:

```bash
cp .env.example .env
# Edit .env and add your credentials
```

### 4. Build features

This step processes the raw database and computes all form/stats features. Takes ~1-2 hours on the full dataset (11 years of races).

```bash
python scripts/build_features.py
```

For a quick test with a subset:

```bash
python scripts/build_features.py --limit 100000
```

### 5. Train the models

```bash
python scripts/train_model.py
```

This trains win and place models with an 18-month validation holdout and prints AUC / Brier scores.

### 6. Predict today's races

```bash
python scripts/predict_today.py
```

Options:

```bash
python scripts/predict_today.py --date 2026-04-12
python scripts/predict_today.py --course Cheltenham
python scripts/predict_today.py --min-edge 10 --max-odds 20
python scripts/predict_today.py --show-all  # Show full predictions per race
```

## Features used by the model

| Category | Features |
|----------|----------|
| Race context | Distance (furlongs), going (ordinal), field size, race class |
| Runner | Age, weight, draw position, official rating, RPR, top speed |
| Market | Log SP, implied probability (bookmaker signal is strong) |
| Horse form | Win rate overall / at distance / on going / at course, last 3 positions, days since last run |
| Jockey | 90-day win rate and place rate |
| Trainer | 90-day win rate and place rate |

## Output interpretation

```
horse          win_prob  place_prob  fair_win_odds  fair_place_odds  bookie_odds
Dashing Star   18.3%     52.1%       5.5            1.9              7.0
Running Brook  14.1%     43.2%       7.1            2.3              6.0
```

**Value bets table:**

```
horse         bet_type  bookie_odds  fair_odds  model_prob  edge_pct  kelly_stake_pct
Dashing Star  win       7.0          5.5        18.3        27.8      6.2
```

- `edge_pct` > 0 = positive expected value
- `kelly_stake_pct` = recommended stake as % of bankroll (half-Kelly, conservative)

## Project structure

```
equintel/
├── data/                   # gitignored — put raceform.db here
├── models/                 # trained model artifacts (gitignored)
├── src/
│   ├── data_loader.py      # Load/parse SQLite DB
│   ├── features.py         # Feature engineering
│   ├── train.py            # Model training pipeline
│   ├── predict.py          # Inference for a single racecard
│   ├── api_client.py       # The Racing API wrapper
│   └── value.py            # Value bet identification
├── scripts/
│   ├── build_features.py   # CLI: compute feature dataset
│   ├── train_model.py      # CLI: train models
│   └── predict_today.py    # CLI: predict and find value bets
├── requirements.txt
├── .env.example
└── README.md
```

## Notes

- The market (SP) is a strong predictor — the model finds edge by combining it with form features the market doesn't fully price in
- Use half-Kelly or smaller stakes; the Kelly Criterion assumes perfect probability estimates
- This is for research and educational purposes
