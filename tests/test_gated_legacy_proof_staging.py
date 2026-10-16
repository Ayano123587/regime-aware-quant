"""Pure legacy proof staging guards; never read actual model artifacts."""

import pytest
from src import prepare_gated_legacy_proof as staging


def test_dirty_source_fails_before_private_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        staging.subprocess, "check_output", lambda *a, **k: "?? pending.py"
    )

    def forbidden(*_):
        raise AssertionError("Private bytes cannot be read")

    monkeypatch.setattr(staging.legacy, "sha", forbidden)
    out = tmp_path / "fresh"
    with pytest.raises(ValueError, match="Commit"):
        staging.prepare("bundle", "selection", "receipt", out, "owner/private")
    assert not out.exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "receipt").write_text("preserve")

    def forbidden(*a, **k):
        raise AssertionError("No Git or private read")

    monkeypatch.setattr(staging.subprocess, "check_output", forbidden)
    with pytest.raises(ValueError, match="Fresh"):
        staging.prepare("bundle", "selection", "receipt", out, "owner/private")
    assert (out / "receipt").read_text() == "preserve"


def test_wrong_original_artifact_fails_before_output(tmp_path, monkeypatch):
    monkeypatch.setattr(staging.subprocess, "check_output", lambda *a, **k: "")
    calls = []
    monkeypatch.setattr(staging.legacy, "sha", lambda p: calls.append(p) or "wrong")
    out = tmp_path / "fresh"
    with pytest.raises(ValueError, match="original artifact"):
        staging.prepare("bundle", "selection", "receipt", out, "owner/private")
    assert calls == ["bundle"] and not out.exists()
