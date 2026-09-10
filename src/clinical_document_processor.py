"""Clinical document transcription and triage pipeline.

Run from the command line with:

    python clinical_document_processor.py --config /path/to/config.yaml
"""
import argparse


# ==============================================================================
# Imports
# ==============================================================================

# --- Standard library ---
import base64
import io
import json
import logging
import re
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

# --- Third-party (see requirements.txt) ---
import numpy as np
import yaml
import requests
import urllib3
import boto3
from botocore.config import Config as BotoConfig

import fitz                                   # PyMuPDF: PDF rasterization + text-layer probe
import cv2                                     # OpenCV: preprocessing + checkbox detection
from PIL import Image, ImageSequence          # TIFF (multi-frame) handling
# Spark 3.3 is used for Iceberg metadata writes. In CML/CDSW this is usually provided by the Spark runtime.
from pyspark.sql import SparkSession
from pyspark.sql import types as SparkTypes
import cml.data_v1 as cmldata


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class Config:
    # Model endpoint
    model_base_url: str
    model_name: str
    request_timeout_s: int

    # Runtime CDP token command settings for model endpoint authentication
    cdp_workload_name: str
    cdp_profile: str
    cdpcli_verify_tls: bool
    cdpcli_tls_cert_path: Optional[str]

    # Supported inputs controlled by YAML
    supported_document_types: List[str]
    supported_file_extensions: List[str]

    # Per-endpoint TLS configuration
    model_tls_verify: bool
    model_tls_cert_path: Optional[str]
    ozone_tls_verify: bool
    ozone_tls_cert_path: Optional[str]

    # Source and destination stores
    source_backend: str
    source_root: str
    dest_backend: str
    dest_root: str

    # Ozone S3 Gateway
    ozone_endpoint_url: Optional[str]
    ozone_access_key: Optional[str]
    ozone_secret_key: Optional[str]
    ozone_region: str
    ozone_signature_version: str
    ozone_addressing_style: str

    # Metadata sink
    # Use metadata_sink: console, spark_iceberg, or both.
    metadata_sink: str
    spark_data_connection_name: str
    spark_iceberg_table: str
    spark_iceberg_create_table_if_missing: bool
    spark_iceberg_table_location: Optional[str]
    spark_iceberg_bucket_count: int
    spark_iceberg_batch_size: int

    # Resolution and checkbox knobs
    res_judge_px: int
    res_transcribe_px: int
    res_refine_px: int
    checkbox_review_threshold: int
    checkbox_bands: int
    checkbox_band_overlap: float
    checkbox_label_match_threshold: float
    checkbox_debug: bool

    # Page router / hallucination guard
    page_router_enabled: bool
    page_router_debug: bool
    route_dense_checkbox_threshold: int
    route_mixed_checkbox_threshold: int
    transcribe_retry_on_guard: bool
    transcribe_debug: bool
    checkbox_explosion_low_visual_limit: int
    checkbox_explosion_low_visual_md_limit: int
    checkbox_explosion_ratio: float
    checkbox_explosion_abs_margin: int
    repetition_guard_threshold: float
    vlm_temperature: float
    vlm_max_tokens: int

    # Identity resolver
    identity_resolver_enabled: bool
    identity_resolver_max_chars: int
    identity_resolver_max_tokens: int
    identity_resolver_debug: bool

    # Console logging controls
    console_summary_only: bool
    console_progress_every_n_files: int
    console_log_metadata_full: bool

    # Optional pipeline log archival to Ozone
    log_to_ozone: bool
    log_ozone_root: str
    log_file_prefix: str

    # Runtime-only CDP token cache. This is never loaded from YAML.
    api_key: Optional[str] = field(default=None, init=False, repr=False)

    def _tls_verify_arg(self, verify: bool, cert_path: Optional[str]):
        """Return a requests/boto3-compatible TLS verify argument."""
        if not verify:
            return False
        return cert_path or True

    @property
    def model_tls_verify_effective(self) -> bool:
        """TLS verify flag for the model endpoint."""
        return bool(self.model_tls_verify)

    @property
    def model_tls_cert_path_effective(self) -> Optional[str]:
        """CA bundle path for the model endpoint, if configured."""
        return self.model_tls_cert_path

    @property
    def model_tls_verify_arg(self):
        """Value to hand to requests' verify= kwarg for model endpoint calls."""
        return self._tls_verify_arg(
            self.model_tls_verify_effective,
            self.model_tls_cert_path_effective,
        )

    @property
    def ozone_tls_verify_effective(self) -> bool:
        """TLS verify flag for the Ozone S3 Gateway."""
        return bool(self.ozone_tls_verify)

    @property
    def ozone_tls_cert_path_effective(self) -> Optional[str]:
        """CA bundle path for the Ozone S3 Gateway, if configured."""
        return self.ozone_tls_cert_path

    @property
    def ozone_tls_verify_arg(self):
        """Value to hand to boto3's verify= kwarg for Ozone S3 Gateway calls."""
        return self._tls_verify_arg(
            self.ozone_tls_verify_effective,
            self.ozone_tls_cert_path_effective,
        )


def _normalize_extensions(values: List[str]) -> List[str]:
    """Normalize extensions from YAML to lowercase dot-prefixed form."""
    normalized = []
    for value in values or []:
        ext = str(value).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized.append(ext)
    return sorted(set(normalized))


def _normalize_document_types(values: List[str]) -> List[str]:
    """Normalize document type tokens from YAML for filename parsing."""
    return sorted({str(value).strip().upper() for value in (values or []) if str(value).strip()})


def load_config(path: str) -> "Config":
    """Load all runtime configuration from a YAML file."""
    if not path:
        raise ValueError("A YAML config path is required. Example: run_pipeline('config.yaml')")

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    data = yaml.safe_load(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {p}")

    fields = Config.__dataclass_fields__
    init_fields = {name for name, f in fields.items() if f.init}
    unknown = sorted(set(data) - init_fields)
    if unknown:
        raise ValueError(f"Unknown config field(s) in {p}: {unknown}")

    missing = sorted(name for name in init_fields if name not in data)
    if missing:
        raise ValueError(f"Missing required config field(s) in {p}: {missing}")

    cfg = Config(**{name: data[name] for name in init_fields})
    cfg.supported_file_extensions = _normalize_extensions(cfg.supported_file_extensions)
    cfg.supported_document_types = _normalize_document_types(cfg.supported_document_types)

    if not cfg.supported_file_extensions:
        raise ValueError("supported_file_extensions must contain at least one extension")
    if not cfg.supported_document_types:
        raise ValueError("supported_document_types must contain at least one document type")

    cfg.api_key = None
    return cfg


# ==============================================================================
# Logging
# ==============================================================================

class _SuppressOzoneOnlyFilter(logging.Filter):
    """Hide records intended only for the persistent file/Ozone log from console."""
    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, "ozone_only", False)


class _ConsoleSummaryOnlyFilter(logging.Filter):
    """
    Console-safe filter for large runs.

    When attached to the console handler, INFO-level detail logs are hidden unless the
    record is explicitly marked console_always=True. WARNING/ERROR records still show.
    File-backed log handlers do not get this filter, so the persistent Ozone log remains detailed.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return bool(getattr(record, "console_always", False))


def get_logger(name: str = "clindoc") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        h.addFilter(_SuppressOzoneOnlyFilter())
        h._pipeline_console_handler = True
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _console_handlers(log: logging.Logger) -> List[logging.Handler]:
    return [h for h in log.handlers if getattr(h, "_pipeline_console_handler", False)]


def enable_console_summary_only(log: logging.Logger) -> List[Tuple[logging.Handler, logging.Filter]]:
    """Attach summary-only filters to console handlers and return them for cleanup."""
    attached = []
    for h in _console_handlers(log):
        # Avoid stacking duplicate summary filters across interactive reruns.
        if any(isinstance(f, _ConsoleSummaryOnlyFilter) for f in h.filters):
            continue
        f = _ConsoleSummaryOnlyFilter()
        h.addFilter(f)
        attached.append((h, f))
    return attached


def disable_console_summary_only(attached: List[Tuple[logging.Handler, logging.Filter]]) -> None:
    """Remove filters that were attached by enable_console_summary_only()."""
    for h, f in attached:
        try:
            h.removeFilter(f)
        except Exception:
            pass


def log_console_info(log: logging.Logger, message: str, *args) -> None:
    """INFO log that is allowed through even when console_summary_only=True."""
    log.info(message, *args, extra={"console_always": True})


# ==============================================================================
# Filename parsing
# ==============================================================================

def get_supported_exts(cfg: "Config") -> set:
    """Return configured supported file extensions as a set."""
    return set(_normalize_extensions(cfg.supported_file_extensions))


def get_supported_doc_types(cfg: "Config") -> set:
    """Return configured supported document-type tokens as a set."""
    return set(_normalize_document_types(cfg.supported_document_types))


@dataclass
class DocRef:
    file_name: str            # unique identifier (basename)
    source_location: str      # full source path / key
    mrn: str                  # filename-derived MRN used as identity resolver input
    document_type: str
    ext: str


def parse_filename(location: str, cfg: "Config") -> DocRef:
    name = PurePosixPath(location).name
    stem = PurePosixPath(name).stem
    ext = PurePosixPath(name).suffix.lower()
    parts = stem.split("_")

    mrn = parts[0] if parts else ""
    doc_type = "UNKNOWN"
    supported_doc_types = get_supported_doc_types(cfg)
    for tok in parts:
        if tok.upper() in supported_doc_types:
            doc_type = tok.upper()
            break

    return DocRef(
        file_name=name,
        source_location=location,
        mrn=mrn,
        document_type=doc_type,
        ext=ext,
    )


# ==============================================================================
# Storage
# ==============================================================================

class FileStore:
    def list(self, *, exts: set) -> List[str]:
        raise NotImplementedError
    def read(self, location: str) -> bytes:
        raise NotImplementedError
    def write(self, location: str, data: bytes) -> str:
        raise NotImplementedError
    def uri(self, location: str) -> str:
        """Return a user-facing location for metadata."""
        return str(location)


class LocalFileStore(FileStore):
    def __init__(self, root: str):
        self.root = Path(root)

    def _is_hidden(self, p: Path) -> bool:
        """True if any path component below the root starts with '.' (e.g. .ipynb_checkpoints)."""
        try:
            rel = p.relative_to(self.root)
        except ValueError:
            rel = p
        return any(part.startswith(".") for part in rel.parts)

    def list(self, *, exts: set) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(str(p) for p in self.root.rglob("*")
                      if p.is_file() and p.suffix.lower() in exts and not self._is_hidden(p))

    def read(self, location: str) -> bytes:
        return Path(location).read_bytes()

    def write(self, location: str, data: bytes) -> str:
        p = self.root / location if not Path(location).is_absolute() else Path(location)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)

    def uri(self, location: str) -> str:
        return str(location)


class OzoneFileStore(FileStore):
    """S3-compatible access to Apache Ozone via its S3 Gateway.

    root is 'bucket/optional/prefix'. For example:
        source_root: clinical-raw/incoming
        dest_root:   clinical-processed/markdown

    Uses explicit S3 v4 signing and path-style addressing, which avoids common
    AuthorizationHeaderMalformed errors against Ozone S3G.
    """
    def __init__(self, cfg: "Config", root: str):
        self.cfg = cfg
        bucket, _, prefix = root.partition("/")
        self.bucket = bucket.strip()
        self.prefix = prefix.strip("/")

        if not self.bucket:
            raise ValueError("Ozone root must be in the format 'bucket/optional/prefix'")

        if not cfg.ozone_endpoint_url:
            raise ValueError("ozone_endpoint_url is required when using ozone backend")

        if not cfg.ozone_access_key or not cfg.ozone_secret_key:
            raise ValueError(
                "ozone_access_key and ozone_secret_key are required when using ozone backend"
            )

        endpoint = cfg.ozone_endpoint_url.rstrip("/")
        if "/" in endpoint.split("://", 1)[-1]:
            get_logger().warning(
                "ozone_endpoint_url appears to include a path. It should be only the "
                "S3 Gateway endpoint, not bucket/prefix: %s",
                cfg.ozone_endpoint_url,
            )

        get_logger().info(
            "Ozone store configured: endpoint=%s bucket=%s prefix=%s region=%s "
            "signature=%s addressing=%s tls_verify=%s ca_bundle=%s",
            endpoint,
            self.bucket,
            self.prefix or "<bucket-root>",
            cfg.ozone_region or "us-east-1",
            cfg.ozone_signature_version,
            cfg.ozone_addressing_style,
            cfg.ozone_tls_verify_effective,
            cfg.ozone_tls_cert_path_effective or "<system-default>",
        )

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=cfg.ozone_access_key,
            aws_secret_access_key=cfg.ozone_secret_key,
            region_name=cfg.ozone_region or "us-east-1",
            verify=cfg.ozone_tls_verify_arg,
            config=BotoConfig(
                signature_version=cfg.ozone_signature_version or "s3v4",
                s3={"addressing_style": cfg.ozone_addressing_style or "path"},
            ),
        )

    def _key(self, location: str) -> str:
        """Return an object key under this store's configured prefix without double-prefixing."""
        loc = str(location).strip("/")
        if not self.prefix:
            return loc
        if loc == self.prefix or loc.startswith(self.prefix + "/"):
            return loc
        return f"{self.prefix}/{loc}".strip("/")

    def list(self, *, exts: set) -> List[str]:
        keys, token = [], None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": self.prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self.s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                k = obj["Key"]
                parts = PurePosixPath(k).parts
                if PurePosixPath(k).suffix.lower() in exts and                         not any(seg.startswith(".") for seg in parts):
                    keys.append(k)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return sorted(keys)

    def read(self, location: str) -> bytes:
        key = self._key(location)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def write(self, location: str, data: bytes) -> str:
        key = self._key(location)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def uri(self, location: str) -> str:
        """Return a full s3://bucket/key URI for metadata."""
        key = self._key(location)
        return f"s3://{self.bucket}/{key}"


def make_store(cfg: "Config", which: str) -> FileStore:
    backend = getattr(cfg, f"{which}_backend")
    root = getattr(cfg, f"{which}_root")
    if backend == "ozone":
        return OzoneFileStore(cfg, root)
    return LocalFileStore(root)




def attach_pipeline_log_capture_to_file() -> Tuple[logging.Handler, str]:
    """
    Attach a file-backed log handler for this run.

    This does not replace console logging. It adds a second handler so detailed
    pipeline logs are streamed to a local temp file instead of kept in memory.
    At the end of run_pipeline(), that temp file can be uploaded to Ozone.
    """
    log = get_logger()

    # Defensive cleanup for interactive reruns if a prior capture handler was left behind.
    for h in list(log.handlers):
        if getattr(h, "_pipeline_log_capture", False):
            log.removeHandler(h)
            h.close()

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".log",
        prefix="data_extraction_log_",
        delete=False,
    )
    log_file_path = tmp.name
    tmp.close()

    handler = logging.FileHandler(log_file_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    handler._pipeline_log_capture = True
    log.addHandler(handler)
    return handler, log_file_path


def write_pipeline_log_file_to_ozone(cfg: "Config", log_file_path: str) -> Optional[str]:
    """
    Write captured pipeline log file to Ozone when log_to_ozone is enabled.

    The target is configured independently from the document source/destination:
        log_ozone_root: bucket/optional/prefix

    The filename format is:
        <log_file_prefix>_<UTC timestamp>.log
    """
    if not getattr(cfg, "log_to_ozone", False):
        return None

    if not getattr(cfg, "log_ozone_root", ""):
        raise ValueError(
            "log_to_ozone is true, but log_ozone_root is not configured. "
            "Example: log_ozone_root: pipeline-logs/data-extraction-logs"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = cfg.log_file_prefix or "data_extraction_log"
    log_file_name = f"{prefix}_{ts}.log"

    log_store = OzoneFileStore(cfg, cfg.log_ozone_root)
    key = log_store._key(log_file_name)
    with open(log_file_path, "rb") as f:
        log_store.s3.put_object(Bucket=log_store.bucket, Key=key, Body=f)
    return f"s3://{log_store.bucket}/{key}"


# ==============================================================================
# Rendering and preprocessing
# ==============================================================================

def page_count(data: bytes, ext: str) -> int:
    """Number of pages/frames in a PDF or (multi-page) TIFF."""
    ext = ext.lower()
    if ext == ".pdf":
        with fitz.open(stream=data, filetype="pdf") as doc:
            return len(doc)
    else:  # tiff
        with Image.open(io.BytesIO(data)) as im:
            n = getattr(im, "n_frames", None)
            return int(n) if n else sum(1 for _ in ImageSequence.Iterator(im))


def has_text_layer(data: bytes, ext: str, min_chars: int = 40) -> bool:
    """Digital-native probe: True if the PDF already carries an extractable text layer."""
    if ext.lower() != ".pdf":
        return False
    with fitz.open(stream=data, filetype="pdf") as doc:
        chars = sum(len(page.get_text("text").strip()) for page in doc)
    return chars >= min_chars


def render_page(data: bytes, ext: str, page_index: int, target_px: int) -> np.ndarray:
    """Return an RGB uint8 array with long edge == target_px."""
    ext = ext.lower()
    if ext == ".pdf":
        with fitz.open(stream=data, filetype="pdf") as doc:
            page = doc[page_index]
            rect = page.rect
            scale = target_px / max(rect.width, rect.height)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = img[:, :, :3]
            elif pix.n == 1:
                img = np.repeat(img, 3, axis=2)
            return np.ascontiguousarray(img)
    else:  # tiff
        with Image.open(io.BytesIO(data)) as im:
            n_frames = int(getattr(im, "n_frames", 1) or 1)
            if page_index < 0 or page_index >= n_frames:
                raise IndexError(f"TIFF page index {page_index} out of range for {n_frames} frame(s)")
            # ImageSequence.Iterator reuses the same mutable PIL image object.
            # Materializing it with list(...) can therefore make every entry point
            # at the final frame. Seek directly and copy the selected frame instead.
            im.seek(page_index)
            frame = im.convert("RGB").copy()
            w, h = frame.size
            scale = target_px / max(w, h)
            frame = frame.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            return np.array(frame)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Flat-field normalization + CLAHE. Returns an RGB uint8 array."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Flat-field: estimate the illumination background with a wide Gaussian and divide it out.
    sigma = max(gray.shape) / 30.0
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    background = np.where(background == 0, 1, background)
    flat = cv2.divide(gray, background, scale=255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(flat)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def encode_png_b64(rgb: np.ndarray) -> str:
    """RGB array -> base64 PNG (no alpha)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ==============================================================================
# Checkbox detection
# ==============================================================================

def _dedupe_rects(rects: "List[tuple]", center_tol: float = 0.65) -> "List[tuple]":
    """Deduplicate nested/duplicate checkbox contours.

    OpenCV often finds both the outer and inner border of the same printed square.
    Earlier implementations counted both, which is why one page could show 27 boxes
    even when the rendered markdown contained fewer checkbox lines.
    """
    if not rects:
        return []
    sides = [max(w, h) for x, y, w, h in rects]
    med_side = float(np.median(sides)) if sides else 20.0
    tol = max(3.0, med_side * center_tol)

    # Prefer the larger square when inner/outer contours overlap.
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    keep = []
    for r in rects:
        x, y, w, h = r
        cx, cy = x + w / 2.0, y + h / 2.0
        duplicate = False
        for kx, ky, kw, kh in keep:
            kcx, kcy = kx + kw / 2.0, ky + kh / 2.0
            if abs(cx - kcx) <= tol and abs(cy - kcy) <= tol:
                duplicate = True
                break
        if not duplicate:
            keep.append(r)
    return sorted(keep, key=lambda b: (b[1], b[0]))


def detect_checkbox_boxes(rgb: np.ndarray,
                          min_frac: float = 0.008,
                          max_frac: float = 0.045,
                          dedupe: bool = True) -> "List[tuple]":
    """Pixel rects (x, y, w, h) of checkbox-like square outlines anywhere on the page.

    This is used to find checkbox *regions*, not to make the final checked/unchecked
    decision. Checked boxes may have distorted borders, so they are sometimes missed;
    the region crop around nearby boxes is what lets the VLM see the checked boxes.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    long_edge = max(H, W)
    min_side, max_side = min_frac * long_edge, max_frac * long_edge

    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)
    contours, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if not (0.75 <= w / float(h) <= 1.33):
            continue
        fill = cv2.contourArea(approx) / float(w * h + 1e-6)
        if 0.55 <= fill <= 1.05:
            boxes.append((x, y, w, h))
    return _dedupe_rects(boxes) if dedupe else boxes


def count_checkboxes(rgb: np.ndarray) -> int:
    return len(detect_checkbox_boxes(rgb, dedupe=True))


# ==============================================================================
# Model client and prompts
# ==============================================================================

def get_latest_cdp_token(cfg: "Config") -> str:
    """Generate a fresh CDP workload auth token using the CDP CLI."""
    cmd = [
        "cdp",
        "iam",
        "generate-workload-auth-token",
        "--workload-name",
        cfg.cdp_workload_name,
        "--profile",
        cfg.cdp_profile,
    ]
    if not cfg.cdpcli_verify_tls:
        cmd.append("--no-verify-tls")
    elif cfg.cdpcli_tls_cert_path:
        cmd.extend(["--ca-bundle", cfg.cdpcli_tls_cert_path])

    get_logger().info(
        "Generating CDP workload auth token for workload=%s profile=%s verify_tls=%s ca_bundle=%s",
        cfg.cdp_workload_name,
        cfg.cdp_profile,
        cfg.cdpcli_verify_tls,
        cfg.cdpcli_tls_cert_path or "<default>",
    )

    api_key_process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if api_key_process.returncode != 0:
        raise RuntimeError(
            f"Failed to generate CDP token: {api_key_process.stderr.strip()}"
        )

    token = json.loads(api_key_process.stdout)["token"]
    if not token:
        raise RuntimeError("CDP token response did not contain token")

    get_logger().info("Generated CDP workload auth token successfully")
    return token


def get_auth_headers(cfg: "Config", force_refresh: bool = False) -> dict:
    """Return OpenAI-compatible auth headers using a runtime CDP token.

    The token is generated once and cached in cfg.api_key for this process.
    If a request receives 401, callers pass force_refresh=True to replace it.
    """
    if force_refresh or not getattr(cfg, "api_key", None):
        cfg.api_key = get_latest_cdp_token(cfg)

    return {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }


def call_vlm(cfg: "Config", prompt: str, image_b64: str,
             max_tokens: Optional[int] = None) -> str:
    base = cfg.model_base_url.rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    payload = {
        "model": cfg.model_name,
        "temperature": cfg.vlm_temperature,
        "max_tokens": max_tokens or cfg.vlm_max_tokens,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
    }
    headers = get_auth_headers(cfg, force_refresh=False)
    resp = requests.post(url, json=payload, headers=headers,
                         timeout=cfg.request_timeout_s, verify=cfg.model_tls_verify_arg)

    if resp.status_code == 401:
        get_logger().warning(
            "VLM request returned 401 Unauthorized. Refreshing CDP token and retrying once."
        )
        headers = get_auth_headers(cfg, force_refresh=True)
        resp = requests.post(url, json=payload, headers=headers,
                             timeout=cfg.request_timeout_s, verify=cfg.model_tls_verify_arg)

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _strip_fence(text: str) -> str:
    """Remove a leading/trailing Markdown code fence the model sometimes adds."""
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
    t = re.sub(r"\n?```$", "", t)
    return t.strip()


PROMPT_HANDWRITING_JUDGE = (
    "You are inspecting a scanned clinical document page. Estimate how much "
    "handwritten (not typed/printed) content it contains. Answer with EXACTLY one "
    "word and nothing else:\n"
    "  None  - no handwriting at all\n"
    "  Low   - a few handwritten marks or short fills\n"
    "  Heavy - substantial handwritten passages or dense handwritten fields\n"
    "Answer:"
)

# Base transcription pass — the working prompt (verbatim). Deliberately simple: one
# plain line for circled options, no "not-a-checkbox" elaboration (which kept regressing
# either the circle read or the checkbox rendering).
PROMPT_TRANSCRIBE = (
    "Transcribe this scanned referral form to plain text, exactly as it appears.\n"
    "Read top-to-bottom within each column, left column first, then the right column.\n"
    "\n"
    "- This form has checkboxes (small printed squares next to options). Render EVERY "
    "checkbox as a list item beginning with [ ], regardless of whether it looks "
    "marked \u2014 for example `[ ] Diagnosis`. Do NOT decide whether a box is ticked; "
    "just render every checkbox as [ ].\n"
    "- For circled options (e.g. Legal Sex, Gender Identity), keep only the circled word.\n"
    "- Transcribe all handwriting verbatim. Use [illegible] only for handwriting you "
    "genuinely cannot read. Write a signature as [signature].\n"
    "- Transcribe every printed and handwritten line. Do not summarize, group, or "
    "reformat into tables.\n"
    "\n"
    "Output the transcribed text only \u2014 no preamble, no commentary, no code fences."
)

PROMPT_PAGE_ROUTER = """
Look at this scanned clinical document page and classify its layout.
Do not transcribe the page. Return JSON only.

Return exactly this schema:
{
  "page_kind": "checkbox_form" | "clinical_note" | "table_report" | "mixed" | "unknown",
  "real_checkbox_count_estimate": 0,
  "has_dense_checkboxes": false,
  "has_tables": false,
  "has_narrative_sections": false,
  "recommended_transcription_prompt": "checkbox_form" | "general_document",
  "confidence": 0.0
}

Important rules:
- Count only real visible small printed square checkboxes.
- Do not count table cells, section headings, status words, bullets, or normal text as checkboxes.
- If the page is mainly narrative text or tables, recommend general_document.
- If the page is dominated by real checkbox options, recommend checkbox_form.
- Return JSON only. No markdown fences, comments, or prose.
"""

PROMPT_GENERAL_TRANSCRIBE = (
    "Transcribe this scanned clinical document page faithfully to plain text.\n"
    "This page may be a clinical note, encounter summary, table report, or normal document.\n"
    "\n"
    "Critical rules:\n"
    "- Do NOT invent checkboxes. Use [ ] or [x] only when a real printed checkbox square is visible on the page.\n"
    "- Do NOT turn headings, table rows, status values, or section names into checkboxes.\n"
    "- Preserve headings, paragraphs, and tables as readable plain text or simple markdown tables.\n"
    "- Transcribe all visible printed text and handwriting. Use [illegible] only when visible text or handwriting exists but cannot be read.\n"
    "- Do not write [illegible] for blank fields such as Other, Specify, Age of onset, Type, Location(s), or empty fill-in lines. Preserve blank fields as blank.\n"
    "- Stop when the visible page content ends. Do not repeat sections or continue with guessed content.\n"
    "- Do not summarize. Do not add information that is not visible.\n"
    "\n"
    "Output the transcribed text only — no preamble, no commentary, no code fences."
)

# Checkbox population prompt. Pass 2 already rendered all checkboxes as neutral [ ].
# This pass no longer sends the model the full list of expected labels. That caused a
# bad failure mode where the model returned every expected label as checked. Instead,
# each crop asks for ONLY visibly checked boxes in that crop. Everything not returned
# stays [ ], which is the safer default for clinical forms.
PROMPT_CHECKED_CHECKBOXES = """
You are inspecting a cropped clinical form image.

Return ONLY checkboxes that are visibly checked or marked in this crop.
A checkbox is checked only when the small printed square contains a clear pen mark,
tick, X, fill, or handwriting stroke inside the square or crossing its border.

Empty printed squares are NOT checked.
If a square only has its printed border and no pen mark, do not return it.
If you are uncertain, do not return it.

Return JSON only in this format:
[
  {"section": "Reason for Testing", "label": "Diagnosis"},
  {"section": "Sample Information", "label": "DNA: min.10 ug in low TE buffer (Source: )"}
]

Rules:
- Do not return unchecked checkboxes.
- Do not infer checked status from surrounding text, handwriting, or clinical context.
- Look only at the small square immediately beside the option label.
- Use the nearest visible section heading when possible; otherwise use an empty string.
- Use the exact visible option label text after the checkbox. Do not include [x], [ ], or the square itself.
- Return [] if no checked checkboxes are visible in the crop.
- Return JSON only. No markdown fences, comments, or prose.
"""


# ==============================================================================
# Checkbox-state processing
# ==============================================================================

CHECKBOX_RE = re.compile(r"\[[ xX]\]")
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
# Bracket tokens the VLM may emit for non-checkbox artifacts.
# These are protected so repair logic never turns [Signature], [Illegible], or [Image]
# into markdown checkboxes like [ ] Signature.
NON_CHECKBOX_ARTIFACT_LABELS = {"signature", "illegible", "image"}

# Syntax tokens used by cleanup helpers. These are not clinical labels.
_SPECIAL_TOKENS = {"", "x", *NON_CHECKBOX_ARTIFACT_LABELS}


_BRACKETED_LABEL_LINE_RE = re.compile(r"^\s*\[([^\[\]\n]{2,250})\]\s*$")
_BRACKET_REPAIR_SKIP_PREFIXES = {
    "list of tests",
    "page ",
}


def _bracketed_checkbox_label_candidates(text: str) -> List[Tuple[int, str]]:
    """Return lines that look like VLM-emitted checkbox labels in the bad `[Label]` form."""
    candidates: List[Tuple[int, str]] = []
    for idx, line in enumerate((text or "").splitlines()):
        m = _BRACKETED_LABEL_LINE_RE.match(line)
        if not m:
            continue
        label = m.group(1).strip()
        low = label.lower()
        if not label or low in _SPECIAL_TOKENS:
            continue
        if any(low.startswith(prefix) for prefix in _BRACKET_REPAIR_SKIP_PREFIXES):
            continue
        candidates.append((idx, label))
    return candidates


def repair_bracketed_checkbox_labels_if_needed(
    text: str,
    route: str,
    visual_checkbox_count: int,
    page_num: int,
) -> Tuple[str, int]:
    """Repair a known VLM regression on checkbox-form pages.

    Some model responses render checkbox options as:
        [Abnormal inflammatory response]

    The rest of the pipeline expects checkbox targets as:
        [ ] Abnormal inflammatory response

    This repair is intentionally conservative:
    - It only runs on pages routed as checkbox_form.
    - It only runs when there are visual checkbox anchors.
    - It only runs when the transcription produced zero valid markdown checkbox lines.
    - It only converts whole-line bracketed labels.
    """
    if route != "checkbox_form" or int(visual_checkbox_count or 0) <= 0:
        return text, 0

    _, existing_targets = _base_checkbox_targets(text)
    if existing_targets:
        return text, 0

    candidates = _bracketed_checkbox_label_candidates(text)
    if len(candidates) < 3:
        return text, 0

    candidate_by_line = {idx: label for idx, label in candidates}
    repaired_lines: List[str] = []
    repaired_count = 0

    for idx, line in enumerate((text or "").splitlines()):
        if idx in candidate_by_line:
            repaired_lines.append(f"[ ] {candidate_by_line[idx]}")
            repaired_count += 1
        else:
            repaired_lines.append(line)

    get_logger().warning(
        "  page %d: repaired %d bracketed checkbox label(s) from [Label] to [ ] Label",
        page_num,
        repaired_count,
    )
    return "\n".join(repaired_lines), repaired_count



def clean_unselected_selection_fields(text: str) -> str:
    """Blank out echoed circle-option fields where nothing was actually selected.

    When one option is circled the model returns `Legal Sex: Female`. When NOTHING is
    circled it has nothing to keep and echoes every option: `Legal Sex [Male] [Female]
    [Non-binary/U/X]`. Such a line carries >=2 bracket tokens whose contents are option
    words (not the single-char checkbox/`illegible`/`signature`/`image` tokens), so we
    drop the bracketed options and keep just the label. Real checkbox lines (a single
    `[ ]`/`[x]`) and selected fields (no brackets) are untouched.
    """
    out = []
    for ln in text.splitlines():
        opts = [t for t in _BRACKET_RE.findall(ln) if t.strip().lower() not in _SPECIAL_TOKENS]
        if len(opts) >= 2:
            cleaned = _BRACKET_RE.sub("", ln)
            cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
            out.append(cleaned)
        else:
            out.append(ln)
    return "\n".join(out)


def _checkbox_label(line: str) -> str:
    """Text of a checkbox option, minus the box token and surrounding punctuation."""
    return CHECKBOX_RE.sub("", line, count=1).strip(" .:-\t")


def _looks_like_section_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    low = s.lower()
    return (
        s.endswith(":")
        or "(required)" in low
        or "checklist" in low
        or "indicate reason" in low
        or "sample information" in low
        or "reason for testing" in low
    )


def _base_checkbox_targets(base_text: str):
    """Return markdown lines and checkbox targets with context.

    Context is the nearest preceding section heading. This prevents repeated labels
    like `Other (Specify)` from being matched to the wrong checkbox group.
    """
    lines = base_text.splitlines()
    targets = []
    context = ""
    for line_index, ln in enumerate(lines):
        stripped = ln.strip()
        if CHECKBOX_RE.search(ln):
            targets.append({
                "line_index": line_index,
                "label": _checkbox_label(ln),
                "context": context,
            })
        elif _looks_like_section_heading(stripped):
            context = stripped.rstrip(":")
    return lines, targets


def _strip_json_fence(text: str) -> str:
    """Remove a leading/trailing Markdown code fence the model sometimes adds."""
    return _strip_fence(text).strip()


def _extract_json_array(text: str) -> list:
    """Parse a JSON array from a VLM response, tolerating accidental prose/fences."""
    t = _strip_json_fence(text)
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _normalize_text_for_match(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\[[ xX]\]", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _similarity(a: str, b: str) -> float:
    na, nb = _normalize_text_for_match(a), _normalize_text_for_match(b)
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    # Give a boost when the VLM returns a shortened but unambiguous label.
    if na in nb or nb in na:
        short = min(len(na), len(nb)) / max(len(na), len(nb))
        ratio = max(ratio, 0.80 + 0.20 * short)
    return ratio


def _label_counts(targets: List[dict]) -> Dict[str, int]:
    counts = {}
    for t in targets:
        key = _normalize_text_for_match(t["label"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _match_checked_item(label: str, section: str, targets: List[dict], threshold: float) -> Optional[int]:
    """Map a checked label returned from a crop to a target checkbox line.

    Returns the target index, or None when the match is weak/ambiguous.
    """
    if not label or not targets:
        return None

    counts = _label_counts(targets)
    scored = []
    for idx, t in enumerate(targets):
        label_score = _similarity(label, t["label"])
        section_score = _similarity(section, t.get("context", "")) if section else 0.0
        combined = label_score
        if section:
            combined = max(combined, 0.72 * label_score + 0.28 * section_score)
        scored.append((combined, label_score, section_score, idx))

    scored.sort(reverse=True, key=lambda x: x[0])
    best_combined, best_label, best_section, best_idx = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0

    if best_label < threshold:
        return None

    norm_label = _normalize_text_for_match(label)
    duplicate_label = counts.get(norm_label, 0) > 1

    # Repeated labels such as `Other (Specify)` require section context or a clear margin.
    if duplicate_label and not section and (best_combined - second) < 0.12:
        return None

    # If section is supplied but it disagrees strongly, avoid a bad cross-section match.
    if section and best_section < 0.45 and duplicate_label:
        return None

    return best_idx if best_combined >= threshold else None


def _parse_checked_items(resp: str) -> "List[Tuple[str, str]}":
    """Parse checked-only VLM response into (label, section) pairs."""
    out = []
    for item in _extract_json_array(resp):
        if isinstance(item, str):
            label, section = item, ""
        elif isinstance(item, dict):
            label = item.get("label") or item.get("text") or item.get("option") or item.get("checkbox") or ""
            section = item.get("section") or item.get("heading") or item.get("group") or ""
        else:
            continue
        label = str(label).strip()
        section = str(section).strip()
        if label:
            out.append((label, section))
    return out


def checkbox_region_crops(rgb: np.ndarray, boxes: "List[tuple]",
                          fallback_bands: int = 4,
                          overlap: float = 0.10) -> "List[tuple]":
    """Return crop regions likely to contain checkbox groups.

    We do not need to know whether the page is one-column or two-column. The function
    uses OpenCV-detected checkbox outlines as anchors, groups them by vertical section,
    then expands the crop enough to include nearby checked boxes whose outlines may have
    been missed because the check mark distorted the square.
    """
    H, W = rgb.shape[:2]
    boxes = _dedupe_rects(boxes)

    if not boxes:
        # Safe fallback: few full-width bands. The prompt asks for checked boxes only,
        # so unreturned boxes remain [ ].
        crops = []
        n = max(1, int(fallback_bands))
        band_h = H / n
        ov = int(band_h * overlap)
        for i in range(n):
            y0 = max(0, int(i * band_h) - (ov if i > 0 else 0))
            y1 = min(H, int((i + 1) * band_h) + (ov if i < n - 1 else 0))
            crops.append((0, y0, W, y1 - y0))
        return crops

    sides = [max(w, h) for x, y, w, h in boxes]
    side = float(np.median(sides)) if sides else 22.0
    y_gap = max(260.0, side * 12.0)

    groups = []
    current = []
    last_y = None
    for b in sorted(boxes, key=lambda r: (r[1], r[0])):
        x, y, w, h = b
        if current and last_y is not None and (y - last_y) > y_gap:
            groups.append(current)
            current = []
        current.append(b)
        last_y = y
    if current:
        groups.append(current)

    crops = []
    seen = set()
    for g in groups:
        min_x = min(x for x, y, w, h in g)
        min_y = min(y for x, y, w, h in g)
        max_x = max(x + w for x, y, w, h in g)
        max_y = max(y + h for x, y, w, h in g)

        # Right padding captures option labels. Upward padding is intentionally
        # moderate: large enough to include checked boxes missed just above the first
        # anchor, but not so large that unrelated circled demographics/header fields
        # dominate the crop.
        x0 = max(0, int(min_x - max(side * 4, 100)))
        x1 = min(W, int(max_x + max(side * 40, W * 0.35)))
        y0 = max(0, int(min_y - max(side * 4, 160)))
        y1 = min(H, int(max_y + max(side * 5, 110)))

        rect = (x0, y0, x1 - x0, y1 - y0)
        if rect[2] > 20 and rect[3] > 20 and rect not in seen:
            crops.append(rect)
            seen.add(rect)

    return crops


def populate_checkboxes(cfg: "Config", refine_rgb: np.ndarray, base_text: str,
                        n_bands: int = 4, overlap: float = 0.10) -> str:
    """Populate checkbox states using checked-only section crops.

    Important checked-only behavior:
    - We DO NOT pass the full expected checkbox list to the VLM.
    - We DO NOT accept `checked: true` for every visible/expected label.
    - We ask each crop for checked labels only; everything else remains `[ ]`.

    This removes the failure mode where one permissive crop response turns every
    checkbox in the markdown into `[x]`.
    """
    lines, targets = _base_checkbox_targets(base_text)
    if not targets:
        return base_text

    states: Dict[int, bool] = {i: False for i in range(len(targets))}
    threshold = float(getattr(cfg, "checkbox_label_match_threshold", 0.76))
    log = get_logger()

    boxes = detect_checkbox_boxes(refine_rgb, dedupe=True)
    crops = checkbox_region_crops(
        refine_rgb,
        boxes,
        fallback_bands=n_bands,
        overlap=overlap,
    )

    if getattr(cfg, "checkbox_debug", False):
        log.info("checkbox region detector: %d deduped anchor boxes, %d crop(s)", len(boxes), len(crops))

    for crop_index, (x, y, w, h) in enumerate(crops, start=1):
        crop = refine_rgb[y:y + h, x:x + w]
        if crop.size == 0:
            continue

        try:
            resp = call_vlm(cfg, PROMPT_CHECKED_CHECKBOXES, encode_png_b64(crop), max_tokens=900)
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.warning("checked-checkbox crop failed (%s); boxes in this crop stay unconfirmed", exc)
            continue

        if getattr(cfg, "checkbox_debug", False):
            log.info("checked-checkbox crop %d x=%s y=%s w=%s h=%s response:\n%s", crop_index, x, y, w, h, resp)

        for label, section in _parse_checked_items(resp):
            idx = _match_checked_item(label, section, targets, threshold)
            if idx is None:
                if getattr(cfg, "checkbox_debug", False):
                    log.info("ignored checked checkbox label with weak/ambiguous match: section=%r label=%r", section, label)
                continue
            states[idx] = True

    for pos, t in enumerate(targets):
        tok = "[x]" if states.get(pos, False) else "[ ]"
        line_index = t["line_index"]
        lines[line_index] = CHECKBOX_RE.sub(tok, lines[line_index], count=1)
    return "\n".join(lines)


# ==============================================================================
# Routing and transcription guards
# ==============================================================================


# --- Page routing and hallucination guard helpers ---

def _extract_json_object(text: str) -> dict:
    """Parse a JSON object from a VLM response, tolerating accidental prose/fences."""
    t = _strip_fence(text).strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def route_page_for_transcription(cfg: "Config", page_img: np.ndarray,
                                 visual_checkbox_count: int, page_num: int) -> Tuple[str, dict]:
    """Choose checkbox-form vs general-document transcription from page content.

    The decision uses both model-independent visual checkbox count and an optional VLM
    router. Default route is general_document unless the page gives enough evidence
    that it is a checkbox form.
    """
    log = get_logger()
    router = {}

    if getattr(cfg, "page_router_enabled", True):
        try:
            resp = call_vlm(cfg, PROMPT_PAGE_ROUTER, encode_png_b64(page_img), max_tokens=500)
            router = _extract_json_object(resp)
            if getattr(cfg, "page_router_debug", False):
                log.info("  page %d router raw response: %s", page_num, resp)
        except Exception as exc:
            log.warning("  page %d router failed; falling back to visual checkbox count: %s", page_num, exc)
            router = {}

    kind = str(router.get("page_kind", "unknown")).strip().lower()
    rec = str(router.get("recommended_transcription_prompt", "")).strip().lower()
    est_boxes = _safe_int(router.get("real_checkbox_count_estimate", 0), 0)
    dense = _safe_bool(router.get("has_dense_checkboxes", False))

    checkbox_signal = max(int(visual_checkbox_count), est_boxes)
    dense_threshold = int(getattr(cfg, "route_dense_checkbox_threshold", 25))
    mixed_threshold = int(getattr(cfg, "route_mixed_checkbox_threshold", 8))

    route = "general_document"
    if checkbox_signal >= dense_threshold or dense:
        route = "checkbox_form"
    elif checkbox_signal >= mixed_threshold and (kind in {"checkbox_form", "mixed"} or rec == "checkbox_form"):
        route = "checkbox_form"

    if getattr(cfg, "page_router_debug", False):
        log.info(
            "  page %d route=%s visual_boxes=%d router_est=%d kind=%s rec=%s dense=%s",
            page_num, route, visual_checkbox_count, est_boxes, kind, rec, dense,
        )
    return route, router


def _repetition_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 20:
        return 0.0
    return 1.0 - (len(set(lines)) / max(1, len(lines)))


def _transcription_guard_action(cfg: "Config", text: str, visual_checkbox_count: int) -> Tuple[str, str]:
    """Detect checkbox hallucination / repetition loops after transcription."""
    md_checkbox_count = len(CHECKBOX_RE.findall(text or ""))
    visual = int(visual_checkbox_count or 0)

    low_visual_limit = int(getattr(cfg, "checkbox_explosion_low_visual_limit", 20))
    low_visual_md_limit = int(getattr(cfg, "checkbox_explosion_low_visual_md_limit", 50))
    ratio = float(getattr(cfg, "checkbox_explosion_ratio", 3.0))
    margin = int(getattr(cfg, "checkbox_explosion_abs_margin", 30))
    rep_threshold = float(getattr(cfg, "repetition_guard_threshold", 0.35))

    if visual < low_visual_limit and md_checkbox_count > low_visual_md_limit:
        return "retry_general", (
            f"generated checkbox count {md_checkbox_count} far exceeds visual checkbox evidence {visual}"
        )

    if md_checkbox_count > max(int(visual * ratio), visual + margin):
        return "retry_general", (
            f"generated checkbox count {md_checkbox_count} exceeds visual checkbox evidence {visual}"
        )

    rep = _repetition_ratio(text or "")
    if rep > rep_threshold and len([ln for ln in (text or "").splitlines() if ln.strip()]) > 80:
        return "retry_general", f"possible repetition loop detected: repeated_line_ratio={rep:.2f}"

    return "ok", ""


def _maybe_write_transcribe_debug(cfg: "Config", page_num: int, route: str, suffix: str, text: str) -> None:
    """Best-effort raw transcript debug files for local destinations only."""
    if not getattr(cfg, "transcribe_debug", False):
        return
    if getattr(cfg, "dest_backend", "local") != "local":
        return
    try:
        dbg = Path(cfg.dest_root) / "debug"
        dbg.mkdir(parents=True, exist_ok=True)
        (dbg / f"page_{page_num:03d}_{route}_{suffix}.txt").write_text(text or "", encoding="utf-8")
    except Exception as exc:
        get_logger().warning("failed to write transcription debug file: %s", exc)


def transcribe_page_with_router(cfg: "Config", base_pre: np.ndarray,
                                visual_checkbox_count: int, page_num: int) -> Tuple[str, str, List[str]]:
    """Transcribe a page while preserving the validated V3 extraction path by default.

    Behavior:
    - If page_router_enabled is false, use the original V3 direct checkbox-form transcription path:
      PROMPT_TRANSCRIBE -> clean_unselected_selection_fields -> bracket-label repair.
      No router call and no general-document retry are used in this mode.
    - If page_router_enabled is true, use the V4 router/guard behavior.

    Returns: (text, route_used, review_remarks)
    """
    log = get_logger()
    remarks: List[str] = []

    # V3 baseline path: always use the validated V3 checkbox-oriented transcription prompt.
    # Later infrastructure features still remain available: CDP token auth, Ozone, logging,
    # identity metadata, Spark Iceberg metadata, and bracket-label repair.
    if not getattr(cfg, "page_router_enabled", False):
        route = "checkbox_form"
        raw = _strip_fence(call_vlm(cfg, PROMPT_TRANSCRIBE, encode_png_b64(base_pre)))
        _maybe_write_transcribe_debug(cfg, page_num, "v3_direct", "raw", raw)
        text = clean_unselected_selection_fields(raw)
        text, repaired_count = repair_bracketed_checkbox_labels_if_needed(
            text, route, visual_checkbox_count, page_num
        )
        if repaired_count:
            _maybe_write_transcribe_debug(cfg, page_num, "v3_direct", "after_bracket_repair", text)
        return text, route, remarks

    # V4 optional path: router chooses checkbox-form vs general-document transcription.
    route, _router = route_page_for_transcription(cfg, base_pre, visual_checkbox_count, page_num)

    if route == "checkbox_form":
        raw = _strip_fence(call_vlm(cfg, PROMPT_TRANSCRIBE, encode_png_b64(base_pre)))
        _maybe_write_transcribe_debug(cfg, page_num, route, "raw", raw)
        text = clean_unselected_selection_fields(raw)
        text, repaired_count = repair_bracketed_checkbox_labels_if_needed(
            text, route, visual_checkbox_count, page_num
        )
        if repaired_count:
            _maybe_write_transcribe_debug(cfg, page_num, route, "after_bracket_repair", text)
    else:
        raw = _strip_fence(call_vlm(cfg, PROMPT_GENERAL_TRANSCRIBE, encode_png_b64(base_pre)))
        _maybe_write_transcribe_debug(cfg, page_num, route, "raw", raw)
        text = raw

    action, reason = _transcription_guard_action(cfg, text, visual_checkbox_count)
    if action == "retry_general" and getattr(cfg, "transcribe_retry_on_guard", True):
        remarks.append(f"transcription guard retry on page {page_num}: {reason}")
        log.warning("  page %d transcription guard triggered; retrying with general prompt: %s", page_num, reason)
        raw2 = _strip_fence(call_vlm(cfg, PROMPT_GENERAL_TRANSCRIBE, encode_png_b64(base_pre)))
        _maybe_write_transcribe_debug(cfg, page_num, "general_document", "retry_raw", raw2)
        text = raw2
        route = "general_document_retry"

        action2, reason2 = _transcription_guard_action(cfg, text, visual_checkbox_count)
        if action2 != "ok":
            remarks.append(f"transcription guard still suspicious on page {page_num}: {reason2}")
            log.warning("  page %d transcription still suspicious after retry: %s", page_num, reason2)

    return text, route, remarks


# ==============================================================================
# Identity resolution
# ==============================================================================

# --- Identity resolver: metadata-only post-processing ---
# This resolver runs AFTER page transcription and checkbox population are complete.
# It does not change page text, routing, checkbox prompts, or checkbox selection logic.

PROMPT_IDENTITY_RESOLVER = """
You are extracting patient identity fields from transcribed clinical document text.

Return JSON only:
{
  "patient_name": null,
  "patient_dob": null,
  "patient_mrn": null,
  "confidence": "high" | "medium" | "low",
  "evidence": {
    "patient_name": [],
    "patient_dob": [],
    "patient_mrn": []
  },
  "conflicts": [],
  "review_required": true
}

Rules:
- Extract only values explicitly supported by the text.
- Do not infer missing values.
- Do not use provider, doctor, coordinator, parent, guardian, or signer names as the patient name.
- Patient name may appear as Patient Name, Name, in a repeated page header, or in narrative text.
- Date of birth may appear as DOB, Date of Birth, Date Of Birth, or narrative text such as "born on".
- MRN may appear as MRN, Medical Record Number, Medical Record #, or Patient MRN.
- Extract patient_mrn only when it is explicitly labeled as MRN or Medical Record Number.
- Do NOT use Social Security Number, SSN, Hospital ID, Health Card Number, Provincial Health Card, Order Number, Lab Number, Encounter Number, or Accession Number as MRN.
- Preserve the date format exactly as shown in the document.
- If multiple different patient names, DOBs, or MRNs appear, report them in conflicts and set review_required=true.
- If a value is not found, return null for that value.
- Evidence must be short nearby text snippets that support the extracted value.
"""


def call_text_llm(cfg: "Config", prompt: str, text: str,
                  max_tokens: Optional[int] = None) -> str:
    """OpenAI-compatible text-only call used only for metadata enrichment."""
    base = cfg.model_base_url.rstrip("/")
    url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
    payload = {
        "model": cfg.model_name,
        "temperature": cfg.vlm_temperature,
        "max_tokens": max_tokens or getattr(cfg, "identity_resolver_max_tokens", 800),
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
    }
    headers = get_auth_headers(cfg, force_refresh=False)
    resp = requests.post(url, json=payload, headers=headers,
                         timeout=cfg.request_timeout_s, verify=cfg.model_tls_verify_arg)

    if resp.status_code == 401:
        get_logger().warning(
            "Text LLM request returned 401 Unauthorized. Refreshing CDP token and retrying once."
        )
        headers = get_auth_headers(cfg, force_refresh=True)
        resp = requests.post(url, json=payload, headers=headers,
                             timeout=cfg.request_timeout_s, verify=cfg.model_tls_verify_arg)

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _bounded_identity_text(page_markdowns: List[str], max_chars: int) -> str:
    """Build resolver input from all pages, preserving page markers.

    If a document is too long, keep the beginning and end rather than only the
    beginning. Patient identity is often repeated in page headers/footers.
    """
    combined = []
    for idx, page_text in enumerate(page_markdowns):
        combined.append(f"\n\n--- PAGE {idx + 1} ---\n{page_text or ''}")
    doc_text = "\n".join(combined)

    if max_chars and len(doc_text) > max_chars:
        half = max_chars // 2
        doc_text = doc_text[:half] + "\n\n--- MIDDLE TRUNCATED FOR IDENTITY RESOLVER ---\n\n" + doc_text[-half:]
    return doc_text


def resolve_patient_identity_from_markdown(
    cfg: "Config",
    page_markdowns: List[str],
    filename_mrn: Optional[str] = None,
) -> dict:
    """Resolve patient identity from final page markdowns and return metadata fields.

    Filename MRN wins. Document-text MRN is used only when filename_mrn is missing.
    This function is metadata-only and must not modify extracted page text.
    """
    result = {}
    try:
        max_chars = int(getattr(cfg, "identity_resolver_max_chars", 30000))
        doc_text = _bounded_identity_text(page_markdowns, max_chars)

        user_text = (
            f"Filename MRN: {filename_mrn or '[not available]'}\n"
            "If Filename MRN is available, it will be used as patient_mrn. "
            "Still extract patient_name and patient_dob from the document text.\n\n"
            f"Document text:\n{doc_text}"
        )

        raw = call_text_llm(
            cfg,
            PROMPT_IDENTITY_RESOLVER,
            user_text,
            max_tokens=int(getattr(cfg, "identity_resolver_max_tokens", 800)),
        )
        if getattr(cfg, "identity_resolver_debug", False):
            get_logger().info("identity resolver raw response: %s", raw)
        result = _extract_json_object(raw)
    except Exception as exc:
        get_logger().warning("identity resolver failed: %s", exc)
        result = {}

    patient_name = result.get("patient_name")
    patient_dob = result.get("patient_dob")

    # Filename MRN wins. Resolver MRN is fallback only.
    if filename_mrn:
        patient_mrn = filename_mrn
        mrn_source = "filename"
    else:
        patient_mrn = result.get("patient_mrn")
        mrn_source = "identity_resolver" if patient_mrn else None

    confidence = result.get("confidence") or "low"
    evidence = result.get("evidence") or {}
    conflicts = result.get("conflicts") or []

    review_required = bool(result.get("review_required", False))
    if not patient_name or not patient_dob or not patient_mrn or conflicts:
        review_required = True

    return {
        "patient_name": patient_name,
        "patient_dob": patient_dob,
        "patient_mrn": patient_mrn,
        "patient_mrn_source": mrn_source,
        "patient_identity_confidence": confidence,
        "patient_identity_review_required": review_required,
        "patient_identity_evidence": evidence,
        "patient_identity_conflicts": conflicts,
    }


# ==============================================================================
# Page and document processing
# ==============================================================================

@dataclass
class PageResult:
    page_index: int
    handwriting: str           # None | Low | Heavy
    checkbox_count: int
    text: str
    remarks: List[str] = field(default_factory=list)


def _clean_judge(raw: str) -> str:
    t = raw.strip().lower()
    if "heavy" in t:
        return "Heavy"
    if "low" in t:
        return "Low"
    return "None"


def process_page(cfg: "Config", data: bytes, ext: str, page_index: int) -> PageResult:
    log = get_logger()
    n = page_index + 1

    # --- Pass 1: handwriting judge (low-res) ---
    judge_img = preprocess(render_page(data, ext, page_index, cfg.res_judge_px))
    handwriting = _clean_judge(
        call_vlm(cfg, PROMPT_HANDWRITING_JUDGE, encode_png_b64(judge_img), max_tokens=8))
    log.info("  page %d: handwriting=%s", n, handwriting)

    # --- OpenCV checkbox anchor count (model-independent, used for crop discovery) ---
    base_rgb = render_page(data, ext, page_index, cfg.res_transcribe_px)
    base_pre = preprocess(base_rgb)
    checkbox_anchor_count = count_checkboxes(base_pre)
    log.info("  page %d: checkbox anchors=%d", n, checkbox_anchor_count)

    # --- Pass 2: routed base transcription @ 2200 px ---
    # Choose between the checkbox-form prompt and a general
    # clinical-document prompt based on page content. The checked-only checkbox
    # population logic below is unchanged.
    text, transcription_route, transcription_remarks = transcribe_page_with_router(
        cfg, base_pre, checkbox_anchor_count, n)
    log.info("  page %d: transcription_route=%s", n, transcription_route)

    # Use the markdown checkbox line count for review metadata because OpenCV anchors
    # can miss checked boxes and can also double-count inner/outer contours.
    _, checkbox_targets = _base_checkbox_targets(text)
    if (transcription_route == "checkbox_form" and checkbox_anchor_count > 0
            and len(checkbox_targets) == 0):
        log.warning(
            "  page %d: checkbox_form route with %d OpenCV anchors but 0 valid markdown checkbox lines",
            n,
            checkbox_anchor_count,
        )
    if transcription_route.startswith("general_document"):
        # For general notes/tables, do not let OpenCV false-positive square-like shapes
        # become reported checkboxes. Trust markdown targets produced by the general
        # prompt, which is instructed not to invent [ ] tokens.
        checkbox_count = len(checkbox_targets)
    else:
        checkbox_count = max(checkbox_anchor_count, len(checkbox_targets))
    log.info("  page %d: checkbox lines=%d final_count=%d", n, len(checkbox_targets), checkbox_count)

    # --- Pass 3: checkbox population via checked-only section crops ---
    if checkbox_count > 0 and len(checkbox_targets) > 0:
        # RAW high-res render (NO flat-field/CLAHE) for the checkbox pass. Contrast
        # enhancement before VLM inference can smear faint marks into adjacent rows.
        refine_rgb = render_page(data, ext, page_index, cfg.res_refine_px)
        text = populate_checkboxes(cfg, refine_rgb, text,
                                   n_bands=cfg.checkbox_bands, overlap=cfg.checkbox_band_overlap)
        log.info("  page %d: checkboxes populated (checked-only section crop pass)", n)

    # (Handwriting is read verbatim inline by Pass 2; heavy handwriting is flagged for
    #  review below rather than re-read.)

    # --- Per-page review remarks ---
    remarks = list(transcription_remarks)
    if handwriting == "Heavy":
        remarks.append(f"heavy handwriting in page {n}")
    if checkbox_count > cfg.checkbox_review_threshold:
        remarks.append(f"heavy checkboxes in page {n}")
    if "[illegible]" in text:
        remarks.append(f"illegible tags in page {n}")

    return PageResult(page_index=page_index, handwriting=handwriting,
                      checkbox_count=checkbox_count, text=text, remarks=remarks)


def _dest_location(cfg: "Config", ref: "DocRef") -> str:
    """Transcription output key/path (Markdown), derived from the source basename."""
    out_name = f"{PurePosixPath(ref.file_name).stem}.md"
    if cfg.dest_backend == "ozone":
        prefix = cfg.dest_root.split("/", 1)[1] if "/" in cfg.dest_root else ""
        return f"{prefix}/{out_name}".strip("/")
    return out_name


def build_markdown(ref: "DocRef", pages: "List[PageResult]") -> str:
    """Render the document as Markdown text only.

    This function deliberately does not write YAML metadata front matter into the
    Markdown file. Metadata is emitted separately and should be treated as the
    source of truth through the metadata sink, such as Iceberg and/or logs.
    """
    body = [f"# {ref.file_name}", ""]
    for p in pages:
        body.append(f"## Page {p.page_index + 1}")
        body.append("")
        body.append(f"*handwriting: {p.handwriting} &middot; checkboxes: {p.checkbox_count}*")
        body.append("")
        body.append(p.text.strip() if p.text and p.text.strip() else "_(no content extracted)_")
        body.append("")
    return "\n".join(body).rstrip() + "\n"


def process_document(cfg: "Config", src: FileStore, dst: FileStore,
                     ref: "DocRef") -> dict:
    log = get_logger()
    log.info("Processing %s (mrn=%s type=%s)",
             ref.file_name, ref.mrn, ref.document_type)
    now = datetime.now(timezone.utc)

    data = src.read(ref.source_location)
    n_pages = page_count(data, ref.ext)
    log.info("  %s has %d page(s).", ref.file_name, n_pages)

    # Digital-native fast path note (probe only; still transcribed by the VLM here for a
    # uniform, structured output — a text-layer extraction branch can be added if desired).
    digital_native = has_text_layer(data, ref.ext)
    if digital_native:
        log.info("  %s carries a text layer (digital-native).", ref.file_name)

    pages: List[PageResult] = []
    for pidx in range(n_pages):
        log.info("  -> processing page %d/%d", pidx + 1, n_pages)
        try:
            pages.append(process_page(cfg, data, ref.ext, pidx))
        except Exception as exc:  # isolate a bad page; keep the rest of the document
            log.exception("  page %d failed: %s", pidx + 1, exc)
            pages.append(PageResult(
                page_index=pidx, handwriting="None", checkbox_count=0,
                text="[page processing error]",
                remarks=[f"processing error in page {pidx + 1}"]))

    review_remarks = [r for p in pages for r in p.remarks]
    status = "review" if review_remarks else "ready"

    # Metadata-only identity resolver. This runs after page extraction and does
    # not alter page text, routing, checkbox prompts, or checkbox population.
    identity_metadata = {}
    if getattr(cfg, "identity_resolver_enabled", True):
        identity_metadata = resolve_patient_identity_from_markdown(
            cfg,
            [p.text for p in pages],
            filename_mrn=ref.mrn if ref.mrn else None,
        )

    # Write the transcription as Markdown text only.
    # Metadata stays out of the Markdown file; metadata is emitted separately.
    md_text = build_markdown(ref, pages)
    dest_location = dst.write(_dest_location(cfg, ref), md_text.encode("utf-8"))

    metadata = {
        "document_type": ref.document_type,
        "source_document_name": ref.file_name,
        "source_document_location": src.uri(ref.source_location),
        "destination_file_location": dest_location,
        "num_pages": n_pages,
        "digital_native": digital_native,
        "status": status,
        "review_remarks": review_remarks,
        "created_dt": now,
        "updated_dt": now,
    }
    metadata.update(identity_metadata)
    return metadata


# ==============================================================================
# Metadata sinks
# ==============================================================================

SPARK_ICEBERG_COLUMNS = [
    "source_document_name", "document_type",
    "source_document_location", "destination_file_location", "num_pages",
    "digital_native", "status", "review_remarks",
    "created_dt", "updated_dt",
    "patient_name", "patient_dob", "patient_mrn", "patient_mrn_source",
    "patient_identity_confidence", "patient_identity_review_required",
    "patient_identity_evidence_json", "patient_identity_conflicts_json",
]


def _metadata_printable(metadata: dict) -> dict:
    printable = dict(metadata)
    for k in ("created_dt", "updated_dt"):
        if isinstance(printable.get(k), datetime):
            printable[k] = printable[k].isoformat()
    return printable


def _console_emit(cfg: "Config", metadata: dict) -> None:
    """Emit metadata through the logger without forcing huge JSON into interactive output.

    When console_log_metadata_full=False, the console receives a small summary while
    the file-backed Ozone log receives the full JSON record as an ozone_only message.
    """
    printable = _metadata_printable(metadata)
    log = get_logger()

    if getattr(cfg, "console_log_metadata_full", False):
        log.info("METADATA %s", json.dumps(printable, ensure_ascii=False))
        return

    remarks = printable.get("review_remarks") or []
    log.info(
        "METADATA source_document_name=%s status=%s pages=%s remarks=%d dest=%s",
        printable.get("source_document_name"),
        printable.get("status"),
        printable.get("num_pages"),
        len(remarks),
        printable.get("destination_file_location"),
    )
    # Persist the full metadata details only to file-backed handlers. The console
    # handler suppresses records marked ozone_only=True.
    log.info(
        "METADATA_FULL %s",
        json.dumps(printable, ensure_ascii=False),
        extra={"ozone_only": True},
    )


def _metadata_sink_parts(cfg: "Config") -> set:
    sink = (cfg.metadata_sink or "console").strip().lower()
    if sink == "both":
        return {"console", "spark_iceberg"}
    if sink in {"spark", "iceberg"}:
        return {"spark_iceberg"}
    return {sink}


def _spark_schema() -> "SparkTypes.StructType":
    return SparkTypes.StructType([
        SparkTypes.StructField("source_document_name", SparkTypes.StringType(), False),
        SparkTypes.StructField("document_type", SparkTypes.StringType(), True),
        SparkTypes.StructField("source_document_location", SparkTypes.StringType(), True),
        SparkTypes.StructField("destination_file_location", SparkTypes.StringType(), True),
        SparkTypes.StructField("num_pages", SparkTypes.IntegerType(), True),
        SparkTypes.StructField("digital_native", SparkTypes.BooleanType(), True),
        SparkTypes.StructField("status", SparkTypes.StringType(), True),
        SparkTypes.StructField("review_remarks", SparkTypes.ArrayType(SparkTypes.StringType()), True),
        SparkTypes.StructField("created_dt", SparkTypes.TimestampType(), True),
        SparkTypes.StructField("updated_dt", SparkTypes.TimestampType(), True),
        SparkTypes.StructField("patient_name", SparkTypes.StringType(), True),
        SparkTypes.StructField("patient_dob", SparkTypes.StringType(), True),
        SparkTypes.StructField("patient_mrn", SparkTypes.StringType(), True),
        SparkTypes.StructField("patient_mrn_source", SparkTypes.StringType(), True),
        SparkTypes.StructField("patient_identity_confidence", SparkTypes.StringType(), True),
        SparkTypes.StructField("patient_identity_review_required", SparkTypes.BooleanType(), True),
        SparkTypes.StructField("patient_identity_evidence_json", SparkTypes.StringType(), True),
        SparkTypes.StructField("patient_identity_conflicts_json", SparkTypes.StringType(), True),
    ])


def get_spark_session(cfg: "Config") -> "SparkSession":
    conn = cmldata.get_connection(cfg.spark_data_connection_name)
    spark = conn.get_spark_session()
    if spark is not None:
        return spark
    return SparkSession.builder.getOrCreate()


def _sql_string(value: str) -> str:
    """Single-quote escape a value for simple Spark SQL DDL assembly."""
    return "'" + str(value).replace("'", "''") + "'"


def create_spark_iceberg_metadata_table_if_needed(cfg: "Config") -> None:
    """Create the Iceberg metadata table through Spark if configured.

    The table uses Iceberg format v2 because row-level MERGE/UPDATE/DELETE operations
    require format v2 in typical Spark + Iceberg deployments.
    """
    if not getattr(cfg, "spark_iceberg_create_table_if_missing", True):
        return

    table = cfg.spark_iceberg_table
    bucket_count = int(getattr(cfg, "spark_iceberg_bucket_count", 64) or 64)
    location = getattr(cfg, "spark_iceberg_table_location", None)
    location_sql = f"\nLOCATION {_sql_string(location)}" if location else ""

    ddl = f"""
CREATE TABLE IF NOT EXISTS {table} (
    source_document_name STRING NOT NULL,
    document_type STRING,
    source_document_location STRING,
    destination_file_location STRING,
    num_pages INT,
    digital_native BOOLEAN,
    status STRING,
    review_remarks ARRAY<STRING>,
    created_dt TIMESTAMP,
    updated_dt TIMESTAMP,
    patient_name STRING,
    patient_dob STRING,
    patient_mrn STRING,
    patient_mrn_source STRING,
    patient_identity_confidence STRING,
    patient_identity_review_required BOOLEAN,
    patient_identity_evidence_json STRING,
    patient_identity_conflicts_json STRING
)
USING iceberg
PARTITIONED BY (bucket({bucket_count}, source_document_name)){location_sql}
TBLPROPERTIES (
    'format-version' = '2',
    'write.parquet.compression-codec' = 'zstd'
)
"""
    get_spark_session(cfg).sql(ddl)
    get_logger().info("Spark Iceberg metadata table available: %s", table)


def _metadata_to_spark_row(metadata: dict) -> dict:
    """Convert one metadata dictionary to the Spark/Iceberg row shape."""
    row = {c: metadata.get(c) for c in SPARK_ICEBERG_COLUMNS}
    row["review_remarks"] = metadata.get("review_remarks") or []
    row["patient_identity_evidence_json"] = json.dumps(
        metadata.get("patient_identity_evidence") or {},
        ensure_ascii=False,
    )
    row["patient_identity_conflicts_json"] = json.dumps(
        metadata.get("patient_identity_conflicts") or [],
        ensure_ascii=False,
    )
    for k in ("created_dt", "updated_dt"):
        v = row.get(k)
        if isinstance(v, datetime) and v.tzinfo is not None:
            # Spark TimestampType stores timestamp values without timezone metadata.
            # Convert to UTC-naive so values are stable regardless of local timezone.
            row[k] = v.astimezone(timezone.utc).replace(tzinfo=None)
    return row


def spark_iceberg_merge_batch(cfg: "Config", metadata_batch: "List[dict]") -> None:
    """Merge a batch of metadata rows into Iceberg using Spark SQL."""
    if not metadata_batch:
        return

    create_spark_iceberg_metadata_table_if_needed(cfg)
    spark = get_spark_session(cfg)
    schema = _spark_schema()
    rows = [_metadata_to_spark_row(m) for m in metadata_batch]
    staging_df = spark.createDataFrame(rows, schema=schema)

    view_name = f"_metadata_staging_{int(datetime.now(timezone.utc).timestamp() * 1000000)}"
    staging_df.createOrReplaceTempView(view_name)

    cols = SPARK_ICEBERG_COLUMNS
    update_cols = [c for c in cols if c != "source_document_name"]
    update_set = ",\n        ".join([f"t.{c} = s.{c}" for c in update_cols])
    insert_cols = ", ".join(cols)
    insert_vals = ", ".join([f"s.{c}" for c in cols])

    merge_sql = f"""
MERGE INTO {cfg.spark_iceberg_table} AS t
USING {view_name} AS s
ON t.source_document_name = s.source_document_name
WHEN MATCHED THEN UPDATE SET
        {update_set}
WHEN NOT MATCHED THEN INSERT ({insert_cols})
VALUES ({insert_vals})
"""
    try:
        spark.sql(merge_sql)
        get_logger().info(
            "Spark Iceberg merge ok: table=%s rows=%d",
            cfg.spark_iceberg_table,
            len(metadata_batch),
        )
    finally:
        try:
            spark.catalog.dropTempView(view_name)
        except Exception:
            pass


def emit_metadata(cfg: "Config", metadata: dict) -> None:
    """Emit one metadata record. Used mainly for console-only or small debug paths."""
    sinks = _metadata_sink_parts(cfg)
    if "console" in sinks:
        _console_emit(cfg, metadata)
    if "spark_iceberg" in sinks:
        spark_iceberg_merge_batch(cfg, [metadata])


def emit_metadata_batch(cfg: "Config", metadata_batch: "List[dict]") -> None:
    """Emit a batch of metadata rows to the configured non-console metadata sink."""
    if not metadata_batch:
        return
    sinks = _metadata_sink_parts(cfg)
    if "spark_iceberg" in sinks:
        spark_iceberg_merge_batch(cfg, metadata_batch)


# ==============================================================================
# Batch orchestration
# ==============================================================================

def run_pipeline(config_path: str) -> List[dict]:
    cfg = load_config(config_path)
    log = get_logger()

    capture_handler = None
    capture_log_file_path = None
    console_filter_handles: List[Tuple[logging.Handler, logging.Filter]] = []

    if getattr(cfg, "log_to_ozone", False):
        capture_handler, capture_log_file_path = attach_pipeline_log_capture_to_file()

    if getattr(cfg, "console_summary_only", False):
        console_filter_handles = enable_console_summary_only(log)

    try:
        if not cfg.model_tls_verify_effective or not cfg.ozone_tls_verify_effective:
            # User explicitly disabled verification for one or both HTTPS endpoints;
            # silence repeated urllib3 InsecureRequestWarning messages to keep logs readable.
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        log_console_info(
            log,
            "Config: source=%s:%s dest=%s:%s sink=%s model_tls_verify=%s ozone_tls_verify=%s spark_table=%s log_to_ozone=%s log_ozone_root=%s console_summary_only=%s",
            cfg.source_backend,
            cfg.source_root,
            cfg.dest_backend,
            cfg.dest_root,
            cfg.metadata_sink,
            cfg.model_tls_verify_effective,
            cfg.ozone_tls_verify_effective,
            getattr(cfg, "spark_iceberg_table", "<not set>"),
            cfg.log_to_ozone,
            cfg.log_ozone_root or "<not set>",
            cfg.console_summary_only,
        )

        src = make_store(cfg, "source")
        dst = make_store(cfg, "dest")

        supported_exts = get_supported_exts(cfg)
        locations = src.list(exts=supported_exts)
        total = len(locations)
        log_console_info(log, "Found %d document(s).", total)

        results = []
        metadata_batch: List[dict] = []
        progress_every = int(getattr(cfg, "console_progress_every_n_files", 50) or 0)
        spark_batch_size = int(getattr(cfg, "spark_iceberg_batch_size", 1000) or 1000)
        sinks = _metadata_sink_parts(cfg)

        for idx, loc in enumerate(locations, start=1):
            ref = parse_filename(loc, cfg)
            if ref.ext not in supported_exts:
                log.warning("Skipping unsupported file: %s", loc)
                continue
            try:
                metadata = process_document(cfg, src, dst, ref)
            except Exception as exc:  # keep the batch alive
                log.exception("FAILED %s: %s", ref.file_name, exc)
                now = datetime.now(timezone.utc)
                metadata = {
                    "source_document_name": ref.file_name,
                    "document_type": ref.document_type,
                    "source_document_location": src.uri(ref.source_location),
                    "destination_file_location": None,
                    "num_pages": None,
                    "digital_native": None,
                    "status": "error",
                    "review_remarks": [str(exc)],
                    "created_dt": now,
                    "updated_dt": now,
                    "patient_name": None,
                    "patient_dob": None,
                    "patient_mrn": ref.mrn or None,
                    "patient_mrn_source": "filename" if ref.mrn else None,
                    "patient_identity_confidence": "low",
                    "patient_identity_review_required": True,
                    "patient_identity_evidence": {},
                    "patient_identity_conflicts": [],
                }

            results.append(metadata)

            if "console" in sinks:
                _console_emit(cfg, metadata)

            if "spark_iceberg" in sinks:
                metadata_batch.append(metadata)
                if len(metadata_batch) >= spark_batch_size:
                    emit_metadata_batch(cfg, metadata_batch)
                    metadata_batch.clear()

            if progress_every > 0 and (idx % progress_every == 0 or idx == total):
                ready_so_far = sum(1 for r in results if r.get("status") == "ready")
                review_so_far = sum(1 for r in results if r.get("status") == "review")
                error_so_far = sum(1 for r in results if r.get("status") == "error")
                log_console_info(
                    log,
                    "Progress: processed=%d/%d ready=%d review=%d error=%d pending_metadata_rows=%d",
                    idx,
                    total,
                    ready_so_far,
                    review_so_far,
                    error_so_far,
                    len(metadata_batch),
                )

        if metadata_batch:
            emit_metadata_batch(cfg, metadata_batch)
            metadata_batch.clear()

        ready = sum(1 for r in results if r.get("status") == "ready")
        review = sum(1 for r in results if r.get("status") == "review")
        errors = sum(1 for r in results if r.get("status") == "error")
        log_console_info(log, "Done. ready=%d review=%d error=%d", ready, review, errors)
        return results

    finally:
        # Restore console behavior for future interactive cells/reruns.
        if console_filter_handles:
            disable_console_summary_only(console_filter_handles)

        if capture_handler is not None and capture_log_file_path is not None:
            capture_handler.flush()

            # Remove the file handler before upload/cleanup, so interactive reruns do not
            # accumulate duplicate capture handlers and the file handle is closed.
            log.removeHandler(capture_handler)
            capture_handler.close()

            try:
                log_location = write_pipeline_log_file_to_ozone(cfg, capture_log_file_path)
                if log_location:
                    log.info("Pipeline log written to %s", log_location)
                    try:
                        Path(capture_log_file_path).unlink(missing_ok=True)
                        log.info("Deleted local temp log file: %s", capture_log_file_path)
                    except Exception as cleanup_exc:
                        log.warning(
                            "Could not delete local temp log file %s: %s",
                            capture_log_file_path,
                            cleanup_exc,
                        )
            except Exception as exc:
                # Log upload should not change the extraction result. Keep the local file
                # so it can be inspected or manually uploaded.
                log.error(
                    "Failed to write pipeline log to Ozone. Local temp log retained at %s. Error: %s",
                    capture_log_file_path,
                    exc,
                )


# ==============================================================================
# Command-line entry point
# ==============================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Process clinical PDF and TIFF documents using a YAML "
            "configuration file."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the YAML configuration file.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the pipeline from the command line."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration file does not exist: {config_path}")

    results = run_pipeline(str(config_path))
    print(f"Processed {len(results)} document(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
