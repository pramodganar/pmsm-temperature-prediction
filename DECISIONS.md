# Design decisions

One line of rationale per non-obvious decision. Items marked TODO are not yet
written down anywhere in the repo — fill them in your own words before the
interview; everything else is a summary of rationale that already exists in the
README or notebooks (pointer given).

## Data

- **Drop rows missing `profile_id` instead of imputing** — they can't be
  assigned to a session, so they can't participate in a grouped split (and they
  also lack `i_q`). (notebook 02, `src/data_prep.clean`)
- **Leave `u_d`/`motor_speed` gaps in and impute in-pipeline** — keeps
  prediction able to score records with missing fields instead of refusing.
  (notebook 02)
- **Work with the 2^20-truncated extract rather than re-exporting the full
  1.33M-row table** — a programmatic export now exists
  (`scripts/export_raw.py`, SQLite → CSV with no spreadsheet row limit);
  TODO: state in your own words how the original extract was produced and
  whether re-exporting the full table was possible at the time.
- **`profile_id` is a grouping key, never a feature** — it identifies a session,
  not a physical quantity; using it would memorise sessions. (README Data)

## Split and validation

- **Hold out whole sessions (`GroupShuffleSplit`), not random rows** — lag-1
  autocorrelation ≈ 1.0 means a random split scatters near-duplicates across
  both sides and inflates scores. (README, notebook 01/02)
- **`GroupKFold` for all model selection and tuning** — same leakage argument
  applied inside CV; the contract is tested in `tests/test_predict.py`.
- **Why not a temporal split within sessions instead?** — TODO: your answer
  (e.g. the deployment question is "new motor run", not "future of a known
  run").
- **Every 5th training row for selection/tuning** — consecutive rows are
  nearly identical so a stride subsample loses little information and cuts CV
  cost ~5×; TODO: why 5 specifically and did you sanity-check against the full
  set?
- **RMSE as the selection metric, MAE/R² reported alongside, per-session RMSE
  as the honesty check** — aggregate metrics hide bad runs. (notebook 06)

## Model

- **HistGradientBoostingRegressor** — best grouped-CV and holdout RMSE of the
  shortlist, fits in seconds, handles missing values natively. (`src/model.py`,
  notebook 04)
- **Grid `{lr: [0.05, 0.1], max_iter: [200, 400], max_depth: [None, 8]}`** —
  the winner (`lr=0.05, max_iter=200`) sits on the grid's lower boundary, so
  the grid was extended one step down post-hoc under identical grouped CV:
  `lr=0.025/max_iter=400` scored 12.227 and `lr=0.025/max_iter=800` scored
  12.417, vs 12.207 for the chosen settings — the optimum was not cut off.
- **Ridge alpha=1.0, Lasso alpha=0.001 left untuned** — they are reference
  points, not contenders; with 7 well-conditioned features and ~840k rows
  regularisation has nothing to fix, and at alpha=0.001 the Lasso is
  effectively OLS, which is why both score identically. (documented in
  `src/model.py`)
- **Scaling for linear models only, not trees** — trees are scale-invariant;
  the preprocessor takes `scale=False` for them. (`src/features.py`)
- **Rejected engineered features (i_s, u_s, p_el) despite better holdout** —
  grouped CV showed no gain, and selecting on the holdout would leak it into
  the decision. (notebook 05)
- **No lag/rolling/EWMA features** — the deliverable scores independent
  records (CLI batch, JSON API, app sliders), so a memoryless per-row model
  keeps every serving surface stateless; lag/EWMA features would need the
  caller to supply thermal history and the server to track sessions.
  Published work on this dataset (Kirgsn/Wallscheid) shows EWMA features help,
  so this is the known next step, not an oversight — notebook 06 names
  thermal-history features as the production follow-up.

## Artifact and serving

- **Refit on all 54 sessions for the saved model** — the split exists to
  estimate error, not to ration training data; documented in README Results.
- **Commit `models/model.joblib` and `data/sample/records.csv`** — the repo
  must demo on clone without the ~100 MB raw file. (.gitignore, README)
- **One `predict.predict` behind CLI, Streamlit, and Flask** — a single scoring
  path can't drift from training preprocessing. (`api.py`, `app.py` docstrings)
- **Hardcoded holdout metrics in the app sidebar** — static by design, with a
  comment to update on retrain (`app.py:29-30`); TODO: consider persisting
  metrics next to the artifact instead.
