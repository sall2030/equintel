# Saudi Adaptation — Equintel

Adapting the UK/Ireland XGBoost pipeline for Saudi racing (JCSA — Riyadh King Abdulaziz & Taif King Khalid).

## Data Sources (as of 2026-10)

| Source | Type | Coverage | Access | Notes |
|--------|------|----------|--------|-------|
| [jcsa.sa](https://jcsa.sa/en/racing/) | Official cards + results | Riyadh / Taif seasons | Free PDFs + HTML results | Cards: PDFs under `/globalassets/jcsa/pdf/`. Results: structured pages with times, sectionals, stewards. |
| Issuu (equestrian-jcsa) | Race cards | Current season | Free | Official publisher of bilingual cards. |
| Podium Sports Racing API | Cards, results, markets, form | Saudi Arabia listed | Commercial | Full global coverage including KSA tracks. |
| apihorseracing.com | Results + markets | Limited (22 races noted) | Free tier available | Secondary source. |
| Racing Post / HRN / Wikipedia | Major race results | Saudi Cup etc. | Free | Good for high-profile races only. |

**No free bulk historical SQLite equivalent to the UK Kaggle dataset exists for Saudi.** Primary path: parse official JCSA results pages or PDFs into structured CSV/JSON, then feed the existing feature pipeline.

## Recommended Ingestion Path

1. Download latest season program and racecards from jcsa.sa.
2. Scrape or manually structure past results from `https://jcsa.sa/en/races/` (or use Podium if licensed).
3. Normalize into the same schema expected by `src/data_loader.py` / `src/features.py` (date, horse, jockey, trainer, course, dist_f, going, ran, pos, sp_decimal, etc.).
4. Load via `src/saudi_loader.py` — it automatically calls `add_saudi_features()`.

## Implemented Saudi-Specific Features (`src/saudi_features.py`)

| Feature | Description |
|---------|-------------|
| `surface_dirt` | 1 if dirt, 0 otherwise (default 1 for most JCSA) |
| `is_riyadh` / `is_taif` | Track flags |
| `local_bred_flag` | Local-bred indicator |
| `temperature_c` | Race-day temperature (when available) |
| `horse_saudi_runs` | Prior Saudi starts for the horse |
| `horse_saudi_win_rate` | Expanding win rate on Saudi tracks only |
| `days_since_last_saudi_run` | Days since last Saudi start |
| `international_runner` | Heuristic: fewer than 3 prior Saudi runs |
| `jockey_saudi_win_rate_90d` / `trainer_saudi_win_rate_90d` | Placeholders for 90-day Saudi splits |

These are added automatically when `load_saudi_csv(..., add_features=True)` (default).

## Next Implementation Steps

- Optional scraper for jcsa.sa results pages.
- Full expanding-window rates for jockey/trainer Saudi form.
- Dual-model option: UK base + Saudi fine-tune, or separate Saudi model once enough history accumulates.
- Keep value detection optional (research mode; no public tips).

## Status

Saudi feature layer added and wired into the loader. Historical volume is still lower than UK; start with recent seasons + Saudi Cup history, then expand.
