# RetryDozza

Analisi e modellazione dei flussi pedonali del borgo di Dozza usando dati dei varchi,
dati TIM e dati meteo ARPAE/ERG5.

La run corrente completa usa anche feature evento locali e valuta tre analisi
(`flow`, `nationality`, `age`) su orizzonti di forecast a 1, 3, 6, 12 e 24 ore.
Gli output principali sono in `outputs/slurm_dozza_three_analyses/h*h/` e il
riepilogo multi-orizzonte in `outputs/slurm_dozza_three_analyses/horizon_summary/`.

## Stato della repo e convenzioni

La repo conserva dati sorgente, script, job Slurm, risultati scientifici delle
run principali e il paper. Gli artefatti di test locale non sono mantenuti:
cartelle `*_test`, `*smoketest*`, cache Python e intermedi LaTeX sono
rigenerabili e vanno rimossi prima di sincronizzare o consegnare il progetto.

Nel codice i commenti devono essere brevi, su singola riga e in italiano. Le
eccezioni sono direttive tecniche richieste dagli strumenti, per esempio
shebang, `#SBATCH`, `# noqa` e `# pragma: no cover`.

## Note sui dati meteo ARPAE/ERG5

I file meteo locali sono in `Data/Meteo` e contengono misurazioni orarie per le
celle ERG5 associate all'area di Dozza. Nei CSV locali la colonna temporale e'
`PragaTime`; le colonne numeriche principali sono:

- `TAVG`: temperatura media.
- `PREC`: precipitazione.
- `RHAVG`: umidita' relativa media.
- `RAD`: radiazione/irradianza solare.
- `W_SCAL_INT`: intensita' scalare del vento.
- `W_VEC_DIR`: direzione vettoriale del vento.
- `W_VEC_INT`: intensita' vettoriale del vento.
- `LEAFW`: bagnatura fogliare.
- `ET0`: evapotraspirazione potenziale stimata.

### Interpretazione di `RAD`

Da documentazione ARPAE/ERG5, le variabili meteo simulate includono
l'irradianza solare. Nei nostri CSV questa grandezza e' esposta come `RAD`.
Quindi `RAD` indica la radiazione/irradianza solare globale, non radar e non
precipitazione.

Nel dataset locale i valori di `RAD` sono coerenti con questa interpretazione:
sono pari a `0` nelle ore notturne e raggiungono picchi intorno a `970-982`
nelle ore di massima insolazione. Poiche' ARPAE descrive la variabile come
irradianza solare, l'unita' fisica attesa e' `W/m^2`; questa unita' e' pero'
un'inferenza dalla definizione fisica e dai valori osservati, non una voce
riportata esplicitamente nella pagina HTML ARPAE consultata.

Per il modello predittivo, `RAD` e' utile come proxy di:

- ciclo giorno/notte;
- insolazione;
- nuvolosita' implicita;
- comfort percepito per gli spostamenti pedonali.

Va interpretata con cautela nelle analisi di importanza: una correlazione alta
di `RAD` con ingressi o uscite puo' dipendere anche dal fatto che codifica molto
bene l'ora del giorno e la stagionalita', non solo il meteo in senso stretto.

Nello script `scripts/analyze_dozza_datasets.py`, i nomi colonna vengono
normalizzati e prefissati. Per esempio:

- `RAD` della cella `01584` diventa `meteo_erg5_cell01584_rad`;
- `RAD` della cella `01624` diventa `meteo_erg5_cell01624_rad`;
- vengono create anche feature aggregate come `meteo_erg5_rad_mean`,
  `meteo_erg5_rad_min`, `meteo_erg5_rad_max` e `meteo_erg5_rad_std`.

Fonti:

- ARPAE dataset ERG5: https://dati.arpae.it/it/dataset/erg5-interpolazione-su-griglia-di-dati-meteo
- Esempio d'uso scientifico dei dati ERG5 con `RAD` come global radiation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10672383/

## Analisi predittive

Lo script principale di modellazione e' `scripts/model_dozza_flows.py`. Supporta
tre target set sullo stesso dataset orario unito:

- `flow`: target `ingressi_borgo`, `uscite_borgo`.
- `nationality`: target `tim_Ni_mean15`, `tim_Ns_mean15`, cioe' presenze TIM
  medie orarie di italiani e stranieri.
- `age`: target `tim_F1_mean15` ... `tim_F6_mean15`, cioe' presenze TIM medie
  orarie per fascia d'eta'.

Con `--feature-scope auto`, lo script evita leakage dalla stessa famiglia del
target:

- per i tre target set usa calendario, meteo, TIM e varchi pedonali;
- in `mode=forecast` TIM e varchi dello stesso orario vengono esclusi, mentre
  restano utilizzabili lag e rolling storici;
- le feature della stessa famiglia del target corrente sono escluse se non sono
  lag/rolling esplicitamente richiesti;
- i lag/rolling dei target sono usati solo se richiesti con
  `--include-target-lags` o `--include-target-rolling`.

Ogni analisi salva tre livelli di interpretabilita':

- `model_feature_importance.csv`: importanza intrinseca del miglior modello per
  target, quando disponibile.
- `ablation_study.csv`: contiene sia rimozione leave-one-group-out dei gruppi
  `calendar`, `meteo`, `tim`, `pedoni`, `target_history`, sia scenari cumulativi
  paper-ready come `calendar`, `calendar+meteo`, `calendar+tim`,
  `calendar+pedoni` e full model.
- `single_feature_ablation.csv`: rimozione leave-one-feature-out sulle feature
  selezionate, con riaddestramento del miglior modello feature-based per target
  senza ripetere la ricerca di iperparametri.
- `shap_importance.csv`: SHAP permutation sul miglior modello per target, se
  `--shap-samples` e' maggiore di zero.

Le metriche principali salvate in `model_metrics.csv` sono:

- `mae`, `rmse`, `r2`, `smape_pct`;
- `wape_pct`, piu' stabile della sMAPE quando i target hanno molte ore basse;
- `mase`, errore scalato rispetto a una naive stagionale oraria;
- `peak_mae`, `peak_precision`, `peak_recall`, `peak_f1` per le ore ad alta
  affluenza, definite con soglia al 90-esimo percentile del train.
- `fit_seconds`, `predict_seconds`, `total_seconds`,
  `inference_ms_per_row` per misurare costo di addestramento e inferenza.

Le baseline di confronto includono:

- `dummy_mean`, `dummy_median`;
- `last_hour`, `same_hour_previous_day`, `same_hour_previous_week`;
- `rolling_mean_24h`, `rolling_mean_168h`;
- `sarimax` e `prophet`, disponibili come modelli opzionali se le dipendenze
  sono installate e il budget tempo lo consente;
- `log1p_ridge`, `poisson`, `tweedie` per conteggi non negativi;
- `random_forest`, `extra_trees`, `hist_gradient_boosting`, `xgboost`,
  `lightgbm` per apprendimento tabellare non lineare;
- `two_stage_ridge`, modello a due stadi che separa ore ordinarie e ore ad alta
  affluenza prima della regressione.

Le baseline temporali naive e rolling sono calcolate sulla serie osservata
all'origine della previsione, non sulla serie target gia' traslata
all'orizzonte futuro. In forecast a orizzonte `h`, `last_hour` usa il valore
osservato a `t-1` per prevedere il target a `t+h`; le baseline stagionali usano
i valori osservati a `t-24` e `t-168`. Le rolling terminano prima di `t`, in
coerenza con i lag target dei modelli feature-based.

Per la robustezza temporale, oltre allo split cronologico finale, la run completa
salva:

- `rolling_validation_metrics.csv` e `rolling_validation_summary.csv`;
- `rolling_validation_rank_stability.csv`, che conta in quanti fold mensili ogni
  modello e' risultato il migliore;
- `bootstrap_metric_intervals.csv`, con intervalli empirici al 95% sulle metriche
  del test split finale.

In `mode=forecast`, il default applica anche un embargo temporale pari a
`--horizon-hours` tra train e test, sia nel final split sia nei fold mensili. Il
gap impedisce che le etichette future degli ultimi origin del train cadano nel
periodo valutato.

Ogni analisi salva anche figure esplicative:

- `target_timeseries_train_test.png/.pdf`: andamento dei target con separazione train/test.
- `target_distribution_train_test.png/.pdf`: distribuzione dei target in train e test.
- `model_metric_mae.png/.pdf`, `model_metric_r2.png/.pdf`, `model_metric_smape_pct.png/.pdf`:
  confronto visuale dei modelli.
- `model_metric_wape_pct.png/.pdf`, `model_metric_mase.png/.pdf`,
  `model_metric_peak_f1.png/.pdf`: metriche aggiuntive per pubblicazione.
- `model_metric_fit_seconds.png/.pdf`, `model_metric_inference_ms_per_row.png/.pdf`:
  costo computazionale dei modelli.
- `selected_feature_groups.png/.pdf`: composizione delle feature selezionate per gruppo.
- `ablation_delta_mae.png/.pdf`: impatto della rimozione di ciascun gruppo feature.
- `ablation_group_set_mae.png/.pdf`: confronto degli scenari cumulativi di feature.
- `ablation_single_feature_delta_mae.png/.pdf`: feature singole la cui rimozione
  cambia maggiormente il MAE.
- `permutation_importance_<target>.png/.pdf`, `model_feature_importance_<target>.png/.pdf`,
  `shap_importance_<target>.png/.pdf`: importanza per target.

### Risultati correnti

I risultati multi-orizzonte presenti in `outputs/slurm_dozza_three_analyses/`
sono da rigenerare prima di aggiornare paper e conclusioni. Una versione
precedente calcolava alcune baseline temporali sulla serie target gia'
traslata: a orizzonti maggiori di 1 ora, `last_hour` usava quindi valori futuri
non disponibili all'origine della previsione. Il codice attuale corregge questa
definizione e aggiunge l'embargo temporale; le vecchie metriche con baseline
temporali non vanno usate come risultati paper-ready.

Per generare i job SLURM delle tre analisi:

```bash
python scripts/generate_dozza_slurm_jobs.py
```

Il comando crea tre job separati e `submit_all.sh` nella cartella
`slurm_jobs/dozza_three_analyses`.

Il generatore crea anche `dozza_preprocess.slurm`, che esegue una sola volta il
merge pedoni + TIM + meteo + eventi e salva il dataset condiviso in
`outputs/slurm_dozza_preprocess/dozza_joined_hourly_inner_with_events.csv`. I
job `flow`, `nationality` e `age` partono con dipendenza SLURM `afterok` dal
pre-job, quindi usano tutti lo stesso dataset unito. Per usare invece un CSV
gia' esistente si puo' generare con `--no-preprocess`.

### Eventi locali e cluster

Il cluster non scarica eventi da internet. Il pre-job Slurm usa solo CSV presenti
nel repository:

- `Data/Eventi/manual_events.csv`: eventi manuali di Dozza;
- `Data/Eventi/major_events_2025.csv`: eventi curati nei poli rilevanti;
- `Data/Eventi/downloaded_events_local.csv`: eventi automatici scaricati prima
  in locale e poi congelati sotto `Data/Eventi`.

Per aggiornare il CSV automatico congelato si esegue in locale:

```bash
bash scripts/run_dozza_events_local.sh
```

Lo script locale usa `Data/Eventi/source_config_auto.csv`, salva il report in
`outputs/dozza_events_local/` e aggiorna
`Data/Eventi/downloaded_events_local.csv`. Questo file viene sincronizzato dal
`push` di `send_this.sh`, mentre `outputs/` resta esclusa.

Se un giorno si vuole forzare il download anche nel pre-job Slurm, il generatore
supporta `--download-events-on-cluster`, ma non e' il default.

Per generare una catena senza feature evento bisogna richiederlo esplicitamente
con `--no-events`.

Risorse default dei job generati:

- preprocess: 12G, 30 minuti;
- `flow`: 16G, 2 ore;
- `nationality`: 16G, 2 ore;
- `age`: 24G, 4 ore.

Per default tutte e tre le analisi validano `top-k` sulla griglia
`15,25,30,40,60`. Per rendere `age` piu' leggera in una run esplorativa si puo'
passare `--age-top-k <k>`.

`submit_all.sh` stampa gli ID Slurm e sottomette `flow`, `nationality`, `age`
solo dopo `afterok` del preprocessing. Il job di sintesi multi-orizzonte usa
invece `afterany` sui job modello: parte anche se un singolo modello fallisce e
riepiloga gli output disponibili. `submit_age_only.sh` controlla se il CSV
preprocessato esiste: se manca, crea prima il job di preprocessing e aggancia
`age` con dipendenza `afterok`.

## Pulizia output vecchi

Lo script `scripts/clean_old_analyses.py` prepara una pulizia sicura degli output
di test/smoke, job SLURM obsoleti e duplicati generati per errore. Di default e'
solo una simulazione:

```bash
python scripts/clean_old_analyses.py
```

Per eliminare davvero i candidati mostrati:

```bash
python scripts/clean_old_analyses.py --execute
```

Opzioni utili:

- `--include-failed-current`: include output correnti incompleti, ad esempio un
  target-set fermato prima di `modeling_report.md`.
- `--include-legacy-runs`: include vecchie run modellistiche non-smoketest.
- `--include-logs`: include i log SLURM, mantenendo di default l'ultimo log per
  prefisso job.

Pulizia manuale gia' applicata nella repo corrente:

- rimosse le cartelle `outputs/*_test` e `outputs/*smoketest*`;
- rimossi gli intermedi LaTeX del paper (`main.aux`, `main.bbl`, `main.blg`,
  `main.log`);
- rimosse le cache Python `__pycache__`.
