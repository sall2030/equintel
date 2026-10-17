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
4. Add Saudi-specific flags: `surface` (Dirt/Turf), `local_bred`, `temperature`, `international_runner`.

## Saudi-Specific Feature Ideas

- Surface preference (most Riyadh/Taif racing is dirt).
- Local vs international jockey/trainer performance split.
- Days since last Saudi run vs international.
- Track bias (King Abdulaziz vs King Khalid).
- Prize money relative to class.
- Weather / temperature impact (common in Gulf).

## Next Implementation Steps

- `src/saudi_loader.py` — load structured Saudi CSV/JSON and map to the internal DataFrame.
- Optional scraper for jcsa.sa results pages.
- Dual-model option: UK base + Saudi fine-tune, or separate Saudi model once enough history accumulates.
- Keep value detection optional (research mode; no public tips).

## Status

Skeleton added. Historical volume is lower than UK; start with recent seasons + Saudi Cup history, then expand.
