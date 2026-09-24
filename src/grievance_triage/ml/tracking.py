import hashlib
import logging
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_input_file(path: Path, artifact_subdir: str = "data") -> None:
    mlflow.log_artifact(str(path), artifact_subdir)
    mlflow.set_tag(f"{path.stem}_sha256", file_hash(path))
    mlflow.set_tag(f"{path.stem}_filename", path.name)


def log_config(config) -> None:
    values = {
        key: value for key, value in vars(config).items()
        if isinstance(value, (str, int, float, bool))
    }
    mlflow.log_params(values)