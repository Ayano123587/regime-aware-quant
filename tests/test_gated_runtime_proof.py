"""Cross-runtime proof guards, without weights, arrays or private datasets."""

import pytest
from src.models import gated_runtime_proof as proof


def test_wrong_2025_source_refused_before_model_scan(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="Frozen full 2025"):
        proof.verify(source, tmp_path / "never_scan")


def test_missing_exact_A_B_artifacts_refused_before_model_loading(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.write_bytes(b"fixture")

    class Digest:
        def hexdigest(self):
            return proof.SOURCE

    monkeypatch.setattr(proof.hashlib, "sha256", lambda _: Digest())
    with pytest.raises(ValueError, match="Original A/B"):
        proof.verify(source, tmp_path)
