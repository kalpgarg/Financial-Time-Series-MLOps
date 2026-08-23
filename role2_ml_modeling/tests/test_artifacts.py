"""Contract test for the produced model artifacts.

Guards that the five .pkl files role2 trains stay mutually consistent and match
the interface role3 serving depends on: 95 features, 3 classes, 50 symbols, and
a 768->50 PCA. This is the guarantee that the .pkl inputs/outputs don't drift.

The artifacts are DVC-tracked (not in git), so this whole module is skipped
where they're absent -- e.g. CI without a `dvc pull`. Run `dvc pull` to include
it (that's where it matters: right after training produces them).
"""

from pathlib import Path

import joblib
import pytest

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_REQUIRED = ["feature_columns.pkl", "symbol_encoder.pkl", "finbert_pca.pkl", "xgboost_model.pkl"]

pytestmark = pytest.mark.skipif(
    not all((MODELS_DIR / f).exists() for f in _REQUIRED),
    reason="model artifacts not present (DVC-tracked); run `dvc pull` to enable",
)

N_FEATURES = 95
N_SYMBOLS = 50
N_PCA = 50
FINBERT_DIM = 768


def _load(name):
    return joblib.load(MODELS_DIR / name)


def test_feature_columns_count():
    cols = _load("feature_columns.pkl")
    assert isinstance(cols, list)
    assert len(cols) == N_FEATURES


def test_symbol_encoder_classes():
    le = _load("symbol_encoder.pkl")
    assert len(le.classes_) == N_SYMBOLS


def test_pca_dimensions():
    pca = _load("finbert_pca.pkl")
    assert pca.n_components_ == N_PCA
    assert pca.n_features_in_ == FINBERT_DIM


def test_xgboost_matches_feature_and_class_contract():
    xgb = _load("xgboost_model.pkl")
    cols = _load("feature_columns.pkl")
    # The model must expect exactly the engineered feature set, in 3 classes.
    assert xgb.n_features_in_ == len(cols) == N_FEATURES
    assert list(xgb.classes_) == [0, 1, 2]


def test_embedding_features_match_pca_components():
    cols = _load("feature_columns.pkl")
    pca = _load("finbert_pca.pkl")
    emb_cols = [c for c in cols if c.startswith("emb_")]
    assert len(emb_cols) == pca.n_components_


def test_lightgbm_matches_feature_contract():
    pytest.importorskip("lightgbm")
    if not (MODELS_DIR / "lightgbm_model.pkl").exists():
        pytest.skip("lightgbm_model.pkl not present")
    lgbm = _load("lightgbm_model.pkl")
    assert lgbm.n_features_in_ == N_FEATURES
