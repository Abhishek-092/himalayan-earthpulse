# Himalayan EarthPulse — NISAR radar, measured and explained

NASA Space Apps 2026 · *Dancing with the SARs*. An interactive web app that processes real **NASA-ISRO NISAR**
L-band data (provisional L2 release, Nov 2025 – Nov 2026) to measure and explain surface change in the Indian Himalaya:

| Model | NISAR product | Place | Result (this release) |
|---|---|---|---|
| GDM – ground deformation | GUNW (asc + desc) | Joshimath | settlement LOS velocity +1.9 ± 7.5 / −3.3 ± 7.7 mm/yr — no significant motion in 2025-26 |
| CDM – glacier flow | GOFF + own GCOV feature tracking | Gangotri | ~207 km² glacier system, fastest 5% > 49 m/yr, July speed-up; vs NASA ITS_LIVE r = 0.44 |
| BDM – forest change | GCOV HH/HV, Wishart omnibus test | Joshimath | mostly winter-snow effects (recovering); ~0.2 km² persistent potential disturbance |
| EPITN – interactions | BDM × GDM | Joshimath | no overlap between forest change and reliable ground movement |

## Run
```bash
cd pipeline && uv sync --extra dev && uv run pytest          # 15 synthetic-truth tests
# credentials: pipeline/.env with EARTHDATA_USERNAME / EARTHDATA_PASSWORD (gitignored)
uv run earthpulse discover                                    # NISAR granules for every region
uv run earthpulse resolve --region joshimath                  # auto-detect tracks, models, masks
uv run earthpulse ingest --region joshimath --product GUNW    # remote AOI subsetting (byte ranges)
uv run earthpulse run gdm --region joshimath                  # gdm | cdm | bdm | epitn
uv run earthpulse export                                      # -> web/public/data
cd ../web && npm install && npm run build                     # static site in web/out
```
Add a region: `earthpulse add-region myplace --name "My place" --bbox W S E N` — no code changes.

Methods, limits and AI-use disclosure: see the app's *Methods & limits* page (`web/src/app/about/page.tsx`).

## Preprocessing verification
`uv run earthpulse verify --region <id>` runs measured pass/warn/fail checks on the real inputs of every stage
(calibration convention and value ranges, speckle ENL, common grid, co-registration offsets on stable ground, RTC
residual on opposite slopes, masks, per-granule wavelength, ionosphere, coherence, unwrapping-error repair,
interferogram QC, reference stability, vegetation phase-closure bias, GOFF stable-ground bias, ITS_LIVE agreement).
Results are shown in the app under **Data QC**. Current: Joshimath 20 pass / 2 warn / 0 fail; Gangotri 10 / 2 / 0.
