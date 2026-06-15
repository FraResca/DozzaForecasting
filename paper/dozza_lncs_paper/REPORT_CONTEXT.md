# Paper Context

This folder contains a Springer LNCS-style manuscript generated from the local
RetryDozza project artifacts.

## Current Scope

The paper reports the latest completed multi-horizon modeling outputs in:

- `outputs/slurm_dozza_preprocess`
- `outputs/slurm_dozza_three_analyses/h1h`
- `outputs/slurm_dozza_three_analyses/h3h`
- `outputs/slurm_dozza_three_analyses/h6h`
- `outputs/slurm_dozza_three_analyses/h12h`
- `outputs/slurm_dozza_three_analyses/h24h`
- `outputs/slurm_dozza_three_analyses/horizon_summary`

The reported models use the event-aware joined hourly dataset:

- `outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner_with_events.csv`

The one-hour horizon is used for the most detailed result tables, while the
multi-horizon summary tables and figures report 1, 3, 6, 12, and 24 hour
robustness checks.

## Generated Tables

CSV versions of the manuscript tables are stored in `tables/` and can be
rebuilt with:

```bash
python paper/dozza_lncs_paper/build_tables.py
```

Generated files include:

- `dataset_summary.csv`
- `dataset_profile.csv`
- `tim_dictionary.csv`
- `experimental_setup.csv`
- `best_overall_metrics.csv`
- `best_feature_model_metrics.csv`
- `persistence_comparison.csv`
- `best_model_metrics_by_horizon.csv`
- `best_feature_model_metrics_by_horizon.csv`
- `persistence_comparison_by_horizon.csv`
- `rolling_validation_best.csv`
- `rolling_validation_summary_by_horizon.csv`
- `top_k_selection_by_horizon.csv`
- `primary_flow_model_leaderboard.csv`
- `ablation_leave_one_group_out.csv`
- `selected_feature_groups_one_hour.csv`
- `single_feature_ablation_top.csv`
- `single_feature_ablation_summary.csv`
- `single_feature_ablation_one_hour.csv`
- `selected_feature_group_counts.csv`
- `top_shap_features.csv`
- `event_selected_feature_summary.csv`
- `event_ablation_summary.csv`
- `event_top_shap_features.csv`

Generated paper figures include:

- `flow_target_distribution_paper.pdf/.png`
- `feature_gain_vs_last_hour_by_horizon_flow.pdf/.png`

## Grounded Quantitative Facts

| Item | Value |
| --- | --- |
| Pedestrian raw rows | 77,769 |
| Aggregated pedestrian hourly timestamps | 26,744 |
| Complete pedestrian target rows | 25,036 |
| TIM processed CSV files | 43 |
| TIM raw rows read | 25,561,519 |
| TIM hourly timestamps | 2,883 |
| Weather hourly timestamps | 8,760 |
| Inner joined event-aware modeling rows | 2,735 |
| Event feature columns | 42 |
| Rows with non-zero event signal | 2,132 |
| Train rows | 2,164 |
| Test rows | 541 |
| Forecast horizons | 1, 3, 6, 12, 24 hours |
| Lags | 1, 2, 24, 168 hours |
| Rolling windows | 3, 6, 24 hours |
| Models | Baselines, Ridge/log-Ridge, Poisson, Tweedie, Random Forest, Extra Trees, HGB, XGBoost, LightGBM, two-stage Ridge |

## Result Interpretation

At one hour, Extra Trees has the lowest MAE and WAPE for pedestrian entrances
and exits. Last-hour persistence is best by MAE for all nationality and age
targets across all horizons. Rolling validation is stricter for flow entrances:
it selects last-hour persistence on average, while the final September split
selects Extra Trees by MAE. Event features are selected mainly for nationality
and age targets, especially at 3--12 hour horizons, but leave-one-group-out
ablation is mixed. Leave-one-feature-out ablation shows that the largest
single-feature effects are mostly TIM or target-history lags for TIM targets.
Treat event effects as exploratory predictive associations, not as causal
findings.

## Notes for Final Submission

The manuscript is anonymous-review ready. Before a camera-ready submission, the
author list, affiliations, acknowledgements, and any required data-governance
statements should be filled with the final venue-specific text.
