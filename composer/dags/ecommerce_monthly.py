"""Monthly e-commerce pipeline orchestrator.

Pipeline: GCS monthly CSV  ->  bronze.raw_events partition  ->  Dataform
(silver fct_events, sessions, calendar_dates  ->  gold marts).

Scheduled run: 1st day of every calendar month at 05:00 UTC.
On a scheduled run we load the month that just ended and snapshot for the
first day of the current month.

Manual run: trigger via Airflow UI/API with optional conf payload:

    {"load_month": "2020-10"}

When provided, load_month overrides the scheduled month and snapshot_date is
derived as the first day of the next calendar month.

Required Airflow Variables (Admin -> Variables):
    gcp_project_id          e.g. "ecommerce-project-496110"
    gcs_bucket              e.g. "ecommerce-bucket-csv-files"
    bq_location             e.g. "europe-central2"
    dataform_repository     Dataform repository name (e.g. "ecommerce-pipeline")
    dataform_location       e.g. "europe-central2"
    dataform_git_branch     e.g. "main"
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryInsertJobOperator,
)
from airflow.providers.google.cloud.operators.dataform import (
    DataformCreateCompilationResultOperator,
    DataformCreateWorkflowInvocationOperator,
)
from airflow.providers.google.cloud.sensors.gcs import GCSObjectExistenceSensor

DEFAULT_ARGS = {
    "owner": "data-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ecommerce_monthly",
    description="Monthly load to bronze + Dataform silver/gold refresh.",
    schedule="0 5 1 * *",
    start_date=datetime(2020, 11, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    template_searchpath=["/home/airflow/gcs/dags/sql"],
    tags=["ecommerce", "bronze", "dataform"],
    params={
        "load_month": Param(
            default="",
            type="string",
            pattern=r"^(\d{4}-\d{2})?$",
            description=(
                "Override calendar month to load (YYYY-MM). "
                "Leave empty for a scheduled run (uses the month that just ended)."
            ),
        ),
    },
    render_template_as_native_obj=False,
)
def ecommerce_monthly():
    @task(task_id="compute_run_params")
    def compute_run_params(**context) -> dict:
        params = context["params"]
        override = (params.get("load_month") or "").strip()
        if override:
            load_month = override
        else:
            load_month = context["data_interval_start"].strftime("%Y-%m")

        year, month = (int(part) for part in load_month.split("-"))
        if month == 12:
            snapshot_year, snapshot_month = year + 1, 1
        else:
            snapshot_year, snapshot_month = year, month + 1
        snapshot_date = f"{snapshot_year:04d}-{snapshot_month:02d}-01"

        return {
            "load_month": load_month,
            "snapshot_date": snapshot_date,
            "gcs_object": f"events/month={load_month}/events_{load_month}.csv",
        }

    run_params = compute_run_params()

    wait_for_csv = GCSObjectExistenceSensor(
        task_id="wait_for_monthly_file",
        bucket="{{ var.value.gcs_bucket }}",
        object="{{ ti.xcom_pull(task_ids='compute_run_params')['gcs_object'] }}",
        poke_interval=60,
        timeout=2 * 60 * 60,
        mode="reschedule",
    )

    load_bronze = BigQueryInsertJobOperator(
        task_id="load_bronze_partition",
        configuration={
            "query": {
                "query": "{% include 'load_bronze_partition.sql' %}",
                "useLegacySql": False,
            }
        },
        location="{{ var.value.bq_location }}",
        project_id="{{ var.value.gcp_project_id }}",
    )

    compile_dataform = DataformCreateCompilationResultOperator(
        task_id="compile_dataform",
        project_id="{{ var.value.gcp_project_id }}",
        region="{{ var.value.dataform_location }}",
        repository_id="{{ var.value.dataform_repository }}",
        compilation_result={
            "git_commitish": "{{ var.value.dataform_git_branch }}",
            "code_compilation_config": {
                "vars": {
                    "load_month": "{{ ti.xcom_pull(task_ids='compute_run_params')['load_month'] }}",
                    "snapshot_date": "{{ ti.xcom_pull(task_ids='compute_run_params')['snapshot_date'] }}",
                }
            },
        },
    )

    invoke_dataform = DataformCreateWorkflowInvocationOperator(
        task_id="invoke_dataform",
        project_id="{{ var.value.gcp_project_id }}",
        region="{{ var.value.dataform_location }}",
        repository_id="{{ var.value.dataform_repository }}",
        workflow_invocation={
            "compilation_result": "{{ ti.xcom_pull(task_ids='compile_dataform')['name'] }}",
        },
    )

    run_params >> wait_for_csv >> load_bronze >> compile_dataform >> invoke_dataform


ecommerce_monthly()
