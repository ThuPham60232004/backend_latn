from google.cloud import storage
import config

client = storage.Client()

def upload_to_gcs(file_path, destination_blob_name):
    """Upload file lên Google Cloud Storage."""
    bucket = client.bucket(config.GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    return f"gs://{config.GCS_BUCKET_NAME}/{destination_blob_name}"

