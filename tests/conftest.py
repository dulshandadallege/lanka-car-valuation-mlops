import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_PATH  # noqa: E402
from src.data import build_datasets  # noqa: E402


@pytest.fixture(scope="session")
def datasets():
    return build_datasets()


@pytest.fixture(scope="session")
def bundle():
    if not MODEL_PATH.exists():
        pytest.skip("no trained model - run `python -m src.pipeline` first")
    import joblib
    return joblib.load(MODEL_PATH)
