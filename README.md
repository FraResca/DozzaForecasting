# RetryDozza

Guida operativa per ripetere gli esperimenti Dozza.

## Input richiesti

I dati grezzi non sono versionati. Prima di lanciare la pipeline, ricostruire
questa struttura locale:

```text
Data/
  DozzaPedoni_marzo2022_settembre2025_perS4Cfile_2ott2025 (1).xlsx
  TIM_ENEA/
    <mese>/*.csv.gz
  Meteo/
    <cella>/*.csv
  Eventi/
    city_locations.csv
    manual_events.csv
    major_events_2025.csv
    downloaded_events_local.csv
    source_config_auto.csv
```

Nel repository pubblico resta solo `Data/Eventi/source_config_example.csv`.

## Ambiente

La pipeline e' pensata per Python 3.12 e per l'ambiente conda `s4c` sul cluster.
Le librerie principali sono `pandas`, `numpy`, `scikit-learn`, `matplotlib`,
`joblib`, `openpyxl`, `xgboost`, `lightgbm` e `shap`.

## Eventi locali

Il cluster usa il CSV eventi gia' congelato in `Data/Eventi/downloaded_events_local.csv`.
Per rigenerarlo in locale:

```bash
bash scripts/run_dozza_events_local.sh
```

## Generazione Job Slurm

Generare la catena completa:

```bash
python scripts/generate_dozza_slurm_jobs.py
```

La configurazione predefinita crea:

- un pre-job di preprocessing condiviso;
- 15 job modello: 3 analisi (`flow`, `nationality`, `age`) x 5 orizzonti
  (`1,3,6,12,24`);
- un job finale di sintesi multi-orizzonte.

Output dei job:

```text
slurm_jobs/dozza_three_analyses/
  dozza_preprocess.slurm
  dozza_flow_h1h.slurm ... dozza_flow_h24h.slurm
  dozza_nationality_h1h.slurm ... dozza_nationality_h24h.slurm
  dozza_age_h1h.slurm ... dozza_age_h24h.slurm
  dozza_summary.slurm
  submit_all.sh
```

## Esecuzione Sul Cluster

Sincronizzare codice, job e dati:

```bash
bash send_this.sh dry-push
bash send_this.sh push
bash send_this.sh verify-remote
```

Lanciare la catena completa sul cluster:

```bash
ssh fresca@copernico.unife.it 'cd /hpc/home/fresca/RetryDozza && bash submit_all.sh'
```

Scaricare risultati e log:

```bash
bash send_this.sh pull-results
```

## Esecuzione Locale Leggera

Solo per controllo rapido, non per risultati finali:

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

## Output Attesi

Preprocessing:

```text
outputs/slurm_dozza_preprocess/
  dozza_joined_hourly_inner.csv
  dozza_joined_hourly_inner_with_events.csv
  tim_hourly_features.csv
  weather_hourly_features.csv
  event_hourly_features.csv
```

Modelli:

```text
outputs/slurm_dozza_three_analyses/h<horizon>h/<analysis>/
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
```

Sintesi:

```text
outputs/slurm_dozza_three_analyses/horizon_summary/
```

## Aggiornamento Tabelle Paper

Dopo aver scaricato una run completa:

```bash
python paper/dozza_lncs_paper/build_tables.py
```

Compilazione manuale del paper:

```bash
cd paper/dozza_lncs_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Pulizia

Simulare la pulizia degli output vecchi:

```bash
python scripts/clean_old_analyses.py
```

Eseguire la pulizia:

```bash
python scripts/clean_old_analyses.py --execute
```
