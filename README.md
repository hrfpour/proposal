# Proposal: Reliable Traffic Forecasting (STGNN + Bayesian)

Live site: https://hrfpour.github.io/proposal/  (after phase 4)

## Roadmap (tick as you go)

### Phase 0 — Setup
- [x] Repo created, skeleton pushed (this script)
- [x] Add BasicTS: `git submodule add https://github.com/zezhishao/BasicTS code/BasicTS`
- [x] Tag `v0-skeleton`

### Phase 1 — Literature data (no GPU)
- [x] Fill `data/models.json` (16 models: name, year, venue, paper, new idea, weakness, repo)
- [x] Verify every paper title and repo link by opening it

### Phase 2 — Pipeline works end to end (first GPU use)
- [ ] Colab notebook `00_check` : GPU visible, repo cloned, deps installed
- [ ] Download PEMS04 and PEMS08 (record version)
- [ ] Run ONE baseline for 1-2 epochs -> one `results/*.json` file
- [ ] Full runs: DCRNN, AGCRN, STID (3 seeds each)

### Phase 3 — Minimum proposal prototype
- [ ] AGCRN + Gaussian head (mean, variance) trained with NLL
- [ ] Add MC Dropout at test time
- [ ] Metrics: MAE/RMSE/MAPE + PICP/MPIW/NLL

### Phase 4 — Site
- [ ] `web/` (Next.js static export, same recipe as atlas) reads `data/models.json` and `results/*.json`
- [ ] GitHub Pages workflow
- [ ] Pages: overview, literature, method, results, reproduce, timeline, Q&A

### Phase 5 — Defense
- [ ] Tag `proposal-v1`
- [ ] Table: paper-reported vs reproduced
- [ ] Rehearse Q&A

## Rules
1. No made-up numbers: unfinished runs stay `"status": "pending"`.
2. Same protocol for all models: 12 in -> 12 out, split 6:2:2, z-score.
3. Data and checkpoints never go into git.
