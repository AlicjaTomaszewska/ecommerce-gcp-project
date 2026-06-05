from google.cloud import bigquery
import pandas as pd
from ml.src.config import BQ_QUERY, GCP_PROJECT

def load_data() -> pd.DataFrame:
    """
    Loads features from BigQuery using the BQ_QUERY defined in config.py.
    """
    client = bigquery.Client(project=GCP_PROJECT)
    query_job = client.query(BQ_QUERY)
    df = query_job.to_dataframe()
    return df
