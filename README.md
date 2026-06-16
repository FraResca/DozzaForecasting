# RetryDozza

Reproducibility guide for the Dozza multi-source forecasting experiments.

The repository contains the code used to reproduce the three experimental
tracks reported in the paper:

- `flow`: pedestrian entrances and exits to the historic borough;
- `nationality`: Italian and foreign TIM presence indicators;
- `age`: TIM presence indicators by age band.

Each track is evaluated at five forecast horizons: `1, 3, 6, 12, 24` hours. The
default Slurm workflow runs shared preprocessing once, then launches the 15
track-horizon modeling jobs, and finally builds the multi-horizon summary.

## 1. Restore The Input Data

Raw data is not versioned in the public repository. Before running the
experiments, restore the following local directory structure:

```text
Data/
  DozzaPedoni_marzo2022_settembre2025_perS4Cfile_2ott2025 (1).xlsx
  TIM_ENEA/
    <month>/*.csv.gz
  Meteo/
    <grid-cell>/*.csv
  Eventi/
    city_locations.csv
    manual_events.csv
    major_events_2025.csv
    downloaded_events_local.csv
    source_config_auto.csv
```

Only `Data/Eventi/source_config_example.csv` is kept in the public repository.
The cluster workflow uses the frozen local event file
`Data/Eventi/downloaded_events_local.csv`; it does not download events during
the Slurm run.

## 2. Prepare The Environment

The pipeline was developed for Python 3.12 locally and for the `s4c` conda
environment on the cluster. The main Python dependencies are:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
openpyxl
xgboost
lightgbm
shap
```

## 3. Optional: Regenerate The Local Event Dataset

Run this step only if `Data/Eventi/downloaded_events_local.csv` must be rebuilt
from the configured event sources:

```bash
bash scripts/run_dozza_events_local.sh
```

After this step, sync the resulting `downloaded_events_local.csv` to the
cluster together with the rest of the repository.

## 4. Generate The Slurm Experiment Chain

Generate the full workflow with the default paper configuration:

```bash
python scripts/generate_dozza_slurm_jobs.py
```

This creates:

```text
slurm_jobs/dozza_three_analyses/
  dozza_preprocess.slurm
  dozza_flow_h1h.slurm ... dozza_flow_h24h.slurm
  dozza_nationality_h1h.slurm ... dozza_nationality_h24h.slurm
  dozza_age_h1h.slurm ... dozza_age_h24h.slurm
  dozza_summary.slurm
  submit_all.sh
```

The generated workflow uses:

- one shared preprocessing job;
- causal forecast mode with horizon-specific embargo;
- top-k feature validation for every track and horizon;
- baselines, linear models, tree ensembles, boosting models, and two-stage
  Ridge where configured;
- permutation importance, ablation, single-feature ablation, and SHAP outputs.

## 5. Sync The Repository To The Cluster

Check what would be transferred:

```bash
bash send_this.sh dry-push
```

Push code, job files, and input data:

```bash
bash send_this.sh push
```

Verify that the remote copy contains the expected scripts, jobs, and frozen
event data:

```bash
bash send_this.sh verify-remote
```

The default remote target is:

```text
fresca@copernico.unife.it:/hpc/home/fresca/RetryDozza
```

Override it with `REMOTE_USER`, `REMOTE_HOST`, `REMOTE_BASE`, or `REMOTE_DIR`
if needed.

## 6. Run The Full Experiment Chain

Submit the complete Slurm chain from the local machine:

```bash
ssh fresca@copernico.unife.it 'cd /hpc/home/fresca/RetryDozza && bash submit_all.sh'
```

The dependency structure is:

```text
preprocess
  -> flow h1,h3,h6,h12,h24
  -> nationality h1,h3,h6,h12,h24
  -> age h1,h3,h6,h12,h24
  -> summary
```

The summary job runs after the modeling jobs and collects the available
multi-horizon outputs.

## 7. Download Results And Logs

After the cluster run completes:

```bash
bash send_this.sh pull-results
```

Main result locations:

```text
outputs/slurm_dozza_preprocess/
outputs/slurm_dozza_three_analyses/h1h/
outputs/slurm_dozza_three_analyses/h3h/
outputs/slurm_dozza_three_analyses/h6h/
outputs/slurm_dozza_three_analyses/h12h/
outputs/slurm_dozza_three_analyses/h24h/
outputs/slurm_dozza_three_analyses/horizon_summary/
logs/slurm_dozza/
```

For each `h<horizon>h/<track>/` directory, the expected modeling artifacts are:

```text
model_metrics.csv
best_model_metrics.csv
best_feature_model_metrics.csv
rolling_validation_summary.csv
top_k_auto_selection.csv
ablation_study.csv
single_feature_ablation.csv
permutation_importance.csv
shap_importance.csv
test_predictions.csv
modeling_report.md
modeling_metadata.json
```

## 8. Rebuild Paper Tables And Figures

After downloading a complete run, regenerate the paper tables and figures from
the local outputs:

```bash
python paper/dozza_lncs_paper/build_tables.py
```

Then compile the paper:

```bash
cd paper/dozza_lncs_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The generated paper tables are written to:

```text
paper/dozza_lncs_paper/tables/
```

Paper-ready figures are written to:

```text
paper/dozza_lncs_paper/figures/
```

## 9. Lightweight Local Smoke Test

This command is only for checking that the modeling script runs on a small
subset. It is not the full experiment and should not be used for paper results.
It requires the preprocessed joined dataset to already exist.

```bash
python scripts/model_dozza_flows.py \
  --input-csv outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner_with_events.csv \
  --output-dir /tmp/dozza_smoke \
  --target-set flow \
  --mode forecast \
  --horizon-hours 1 \
  --lags 1,24 \
  --include-target-lags \
  --models dummy_mean,last_hour,ridge \
  --top-k 5 \
  --permutation-repeats 0 \
  --bootstrap-samples 0 \
  --shap-samples 0 \
  --no-ablation \
  --no-single-feature-ablation \
  --max-rows 500 \
  --quick
```

## 10. Cleaning Old Outputs

Preview cleanup of old analysis outputs:

```bash
python scripts/clean_old_analyses.py
```

Run cleanup:

```bash
python scripts/clean_old_analyses.py --execute
```
