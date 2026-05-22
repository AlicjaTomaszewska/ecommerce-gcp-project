# Cloud Composer orchestration

DAG `ecommerce_monthly` orchestrates the monthly pipeline:

```
GCS (events_YYYY-MM.csv)
  -> bronze.raw_events (partition for load_month)
  -> Dataform compile (vars: load_month, snapshot_date)
  -> Dataform workflow invocation (silver + gold)
```

## Files

- `dags/ecommerce_monthly.py` - DAG definition
- `dags/sql/load_bronze_partition.sql` - templated bronze load (called by the DAG)
- `variables_example.json` - Airflow Variables expected by the DAG

## One-time GCP setup

1. **Enable APIs** in the target project: Cloud Composer, BigQuery, Cloud
   Storage, Dataform.
2. **Create a Composer 2 environment** in region `europe-central2`. Default
   service account is fine for a start (note it down; usually
   `<env-name>-sa@<project>.iam.gserviceaccount.com` or the default Compute
   Engine SA).
3. **Grant the Composer service account** these roles on the project:
   - `roles/bigquery.dataEditor`
   - `roles/bigquery.jobUser`
   - `roles/storage.objectViewer` (read CSVs from the source bucket)
   - `roles/dataform.editor` (compile and invoke workflows)
4. **Dataform repository** must already exist (the one with the `definitions/`
   tree from this repo). Note the repository ID and the Git branch you want to
   run from (e.g. `main`).
5. **Source CSV bucket**: upload monthly files to
   `gs://<gcs_bucket>/events/month=YYYY-MM/events_YYYY-MM.csv`. The DAG waits
   for this exact path.

## Configure Airflow Variables

In Composer's Airflow UI -> Admin -> Variables, import
`variables_example.json` or set them manually:

| Variable | Value |
|----------|-------|
| `gcp_project_id` | project ID |
| `gcs_bucket` | bucket with monthly CSVs |
| `bq_location` | `europe-central2` |
| `dataform_repository` | Dataform repository ID |
| `dataform_location` | `europe-central2` |
| `dataform_git_branch` | `main` (or release config branch) |

## Upload the DAG

Composer syncs DAGs from its own GCS bucket. Find it in Composer UI ->
Environment details -> Configuration -> "DAGs folder".

Upload the contents of `composer/dags/` so the layout in the bucket becomes:

```
<composer-dags-bucket>/dags/ecommerce_monthly.py
<composer-dags-bucket>/dags/sql/load_bronze_partition.sql
```

Example with `gsutil`:

```bash
gsutil -m cp -r composer/dags/* gs://<composer-dags-bucket>/dags/
```

After ~30 seconds the DAG should appear in the Airflow UI.

## Running the DAG

### Scheduled (automatic)

Cron: `0 5 1 * *` (5 AM UTC on day 1). On 2021-03-01 the DAG will load
month `2021-02` and snapshot for `2021-03-01`.

`catchup=False` so missed past intervals are NOT re-run automatically; use
the manual mode for backfills.

### Manual / on-demand

Airflow UI -> DAGs -> `ecommerce_monthly` -> Trigger DAG w/ config:

```json
{ "load_month": "2020-10" }
```

This loads October 2020 and snapshots for `2020-11-01`. Leave the field
empty for a normal scheduled-style run (uses `data_interval_start`).

Backfilling all five months sequentially:

```text
2020-10, 2020-11, 2020-12, 2021-01, 2021-02
```

`max_active_runs=1` ensures these run one at a time.

## Task graph

```
compute_run_params
        |
wait_for_monthly_file (GCS sensor, 2 h timeout)
        |
load_bronze_partition (BigQuery DELETE + INSERT for the load_month)
        |
compile_dataform (sets vars: load_month, snapshot_date)
        |
invoke_dataform (synchronous; fails if any action fails)
```

The Dataform invocation runs everything (silver + gold + assertions). To
limit to specific tags or actions, extend the operator with `included_tags`
or `included_targets` in `workflow_invocation`.

## Verifying a run

After a successful run:

```sql
SELECT _load_month, COUNT(*)
FROM `<project>.bronze.raw_events`
GROUP BY _load_month;

SELECT snapshot_date, COUNT(*)
FROM `<project>.gold.product_features_monthly`
GROUP BY snapshot_date;
```

Both should contain a row for the month just processed.
