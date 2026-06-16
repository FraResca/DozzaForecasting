# RetryDozza

Reproducibility guide for the Dozza multi-source forecasting experiments.

The repository contains the code used to reproduce three experimental tracks:

- `flow`: pedestrian entrances and exits to the historic borough;
- `nationality`: Italian and foreign TIM presence indicators;
- `age`: TIM presence indicators by age band.

Each track is evaluated at forecast horizons of `1, 3, 6, 12, 24` hours.

## 1. Restore The Input Data

Raw data is not versioned in the public repository. Before running the
experiments, restore this local directory structure:

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
Use it as a template if the automatic event-source configuration must be
rebuilt.

## 2. Prepare The Environment

The pipeline was developed with Python 3.12. The main Python dependencies are:

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

Install them in a local virtual environment or conda environment before running
the scripts.

## 3. Build The Shared Joined Dataset

Create the hourly pedestrian, TIM, weather, and correlation outputs:

```bash
python scripts/analyze_dozza_datasets.py \
  --data-dir Data \
  --weather-dir Data/Meteo \
  --output-dir outputs/dozza_preprocess \
  --rebuild-tim
```

This step writes the joined hourly datasets and the preprocessing diagnostics to
`outputs/dozza_preprocess/`.

## 4. Add Event Features

If `Data/Eventi/downloaded_events_local.csv` already exists, reuse it to avoid
network-dependent runs. To rebuild event features and join them to the shared
dataset:

```bash
python scripts/build_dozza_event_features.py \
  --manual-events Data/Eventi/manual_events.csv \
  --additional-events-csv Data/Eventi/major_events_2025.csv \
  --additional-events-csv Data/Eventi/downloaded_events_local.csv \
  --city-locations Data/Eventi/city_locations.csv \
  --reference-csv outputs/dozza_preprocess/dozza_joined_hourly_inner.csv \
  --joined-output-csv outputs/dozza_preprocess/dozza_joined_hourly_inner_with_events.csv \
  --extra-joined-reference-csv outputs/dozza_preprocess/dozza_joined_hourly_left.csv \
  --extra-joined-output-csv outputs/dozza_preprocess/dozza_joined_hourly_left_with_events.csv \
  --output-dir outputs/dozza_events
```

To regenerate `downloaded_events_local.csv` from the configured public sources,
run:

```bash
bash scripts/run_dozza_events_local.sh
```

## 5. Run The Forecasting Experiments

The full experiment consists of one shared preprocessing step followed by one
modeling run for each target track and horizon. The commands below reproduce
the full experimental configuration with validated top-k feature selection, baselines,
linear models, tree ensembles, boosting models, permutation importance,
ablation, single-feature ablation, SHAP, and timing metrics.

```bash
INPUT_CSV="outputs/dozza_preprocess/dozza_joined_hourly_inner_with_events.csv"
ROOT_OUT="outputs/dozza_three_analyses"
MODELS="dummy_mean,dummy_median,last_hour,same_hour_previous_day,same_hour_previous_week,rolling_mean_24h,rolling_mean_168h,ridge,log1p_ridge,poisson,tweedie,random_forest,extra_trees,hist_gradient_boosting,xgboost,lightgbm,two_stage_ridge"

for horizon in 1 3 6 12 24; do
  for target_set in flow nationality age; do
    python scripts/model_dozza_flows.py \
      --input-csv "${INPUT_CSV}" \
      --output-dir "${ROOT_OUT}/h${horizon}h/${target_set}" \
      --target-set "${target_set}" \
      --mode forecast \
      --horizon-hours "${horizon}" \
      --lags 1,2,24,168 \
      --rolling-windows 3,6,24 \
      --include-target-lags \
      --include-target-rolling \
      --top-k-grid 15,25,30,40,60 \
      --models "${MODELS}" \
      --rolling-validation \
      --permutation-repeats 10 \
      --bootstrap-samples 500 \
      --shap-samples 300
  done
done
```

For large runs, submit the same Python commands through the execution system
available on your infrastructure. The code does not require a specific
scheduler.

## 6. Build The Multi-Horizon Summary

```bash
python scripts/summarize_dozza_horizon_results.py \
  --root-output-dir outputs/dozza_three_analyses \
  --output-dir outputs/dozza_three_analyses/horizon_summary
```

Expected artifacts for each `h<horizon>h/<track>/` directory include:

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
modeling_metadata.json
```

## 7. Lightweight Local Smoke Test

This command only checks that the modeling script runs on a small configuration.
It is not the full experiment and should not be used for final results.

```bash
python scripts/model_dozza_flows.py \
  --input-csv outputs/dozza_preprocess/dozza_joined_hourly_inner_with_events.csv \
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
  --quick
```
