"""A pinned commit alone does not identify imported working-tree bytes."""
import importlib.util
from pathlib import Path

import pytest


def test_modified_working_file_is_rejected_even_with_matching_commit(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/check_published_alignment.py"
    spec = importlib.util.spec_from_file_location("alignment_check", script)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    revision = "a" * 40
    monkeypatch.setattr(check.subprocess, "check_output", lambda *args, **kwargs: revision)
    target = tmp_path / "model.py"
    target.write_text("value = 1\n")
    pin = {"commit": revision, "file_sha256": {"model.py": check.sha256(target)}}
    assert check.verify_checkout(tmp_path, pin) == tmp_path.resolve()
    target.write_text("value = 2\n")
    with pytest.raises(ValueError, match="Authors source mismatch"):
        check.verify_checkout(tmp_path, pin)
