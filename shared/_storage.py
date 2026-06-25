"""Supabase Storage helpers via S3 protocol (boto3)."""

import os
import re
import unicodedata
from dotenv import load_dotenv

load_dotenv()

BUCKET   = "licitaciones-docs"
DOCS_DIR = "docs"


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("SUPABASE_S3_ENDPOINT"),
        region_name=os.getenv("SUPABASE_S3_REGION", "us-east-2"),
        aws_access_key_id=os.getenv("SUPABASE_S3_KEY_ID"),
        aws_secret_access_key=os.getenv("SUPABASE_S3_SECRET"),
    )


def make_key(licitacion_id: int, tipo: str, ext: str) -> str:
    """Build storage key: {licitacion_id}/{safe_filename}.{ext} (ASCII-safe)."""
    name = _ascii_safe(tipo)
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    filename = (name[:80] or "documento") + f".{ext}"
    return f"{licitacion_id}/{filename}"


def _ascii_safe(text: str) -> str:
    """Normalize accented chars to ASCII equivalents."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def local_to_key(local_path: str) -> str:
    """Convert docs/271717/foo.pdf → 271717/foo.pdf (ASCII-safe)."""
    rel = os.path.relpath(local_path, DOCS_DIR)
    return _ascii_safe(rel).replace("\\", "/")


def upload_bytes(key: str, data: bytes) -> None:
    """Upload raw bytes to the bucket at the given key."""
    import io
    _client().upload_fileobj(io.BytesIO(data), BUCKET, key)


def upload_doc(local_path: str, data: bytes) -> None:
    """Upload using local path to derive the storage key."""
    upload_bytes(local_to_key(local_path), data)


def download_doc(key: str) -> bytes | None:
    """Download file from Storage by key. Returns None if not found."""
    import io
    buf = io.BytesIO()
    try:
        _client().download_fileobj(BUCKET, key, buf)
        return buf.getvalue()
    except Exception:
        return None


def delete_doc(key: str) -> bool:
    """Delete a file from Storage. Returns True on success."""
    try:
        _client().delete_object(Bucket=BUCKET, Key=key)
        return True
    except Exception:
        return False
