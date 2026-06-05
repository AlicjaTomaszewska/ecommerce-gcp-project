import os
import sys
from pathlib import Path
from google.cloud import storage

# Ensure the parent directory is in python path
sys.path.append(str(Path(__file__).resolve().parent))
from src.config import GCP_PROJECT, ARTIFACT_DIR

def upload_model():
    local_model_path = ARTIFACT_DIR / "hurdle_model.joblib"
    if not local_model_path.exists():
        print(f"Error: Local model file not found at {local_model_path}")
        print("Please run the training pipeline first from ml directory:")
        print("   python -m src.training.train_pipeline")
        sys.exit(1)

    bucket_name = f"{GCP_PROJECT}-models"
    blob_name = "demand_forecast_model.joblib"

    print(f"Initializing Storage Client for project: {GCP_PROJECT}...")
    storage_client = storage.Client(project=GCP_PROJECT)

    # Check if bucket exists, create if not
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        print(f"Bucket '{bucket_name}' not found. Creating bucket...")
        # Create bucket in europe-central2 to match the rest of the stack
        bucket = storage_client.create_bucket(bucket_name, location="europe-central2")
        print(f"Created bucket '{bucket_name}'.")

    blob = bucket.blob(blob_name)
    print(f"Uploading {local_model_path} to gs://{bucket_name}/{blob_name}...")
    blob.upload_from_filename(str(local_model_path))
    print("Upload completed successfully!")

if __name__ == "__main__":
    upload_model()
