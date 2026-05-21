# Monthly pipeline run order

1. Load bronze for one month (`02_load_bronze_month.sql`).
2. Set Dataform `vars` in `workflow_settings.yaml` (or release config):
   - `load_month`: e.g. `2020-10`
   - `snapshot_date`: first day of the **next** month, e.g. `2020-11-01`
3. Compile and execute Dataform (silver incremental for that month, gold snapshot for `snapshot_date`).

See `config/monthly_loads.json` for all five full months (September 2020 excluded).

Backfill: repeat steps 1–3 for each entry in `monthly_loads.json`, in order.
