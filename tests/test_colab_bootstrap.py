import importlib.util
import io
from pathlib import Path
import stat
import urllib.request
import zipfile

import pytest


spec = importlib.util.spec_from_file_location("colab_bootstrap", Path(__file__).resolve().parents[1] / "scripts/colab_bootstrap.py")
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
REVISION = "a" * 40


def archive_bytes(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return buffer.getvalue()


def test_public_download_uses_anonymous_pinned_archive(tmp_path, monkeypatch):
    payload = archive_bytes([("repo-revision/", ""), ("repo-revision/src/module.py", "VALUE = 1\n")])
    observed = []

    class Opener:
        def open(self, request, timeout):
            observed.append(request)
            return io.BytesIO(payload)

    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: Opener())
    destination = tmp_path / "source"
    assert bootstrap.fetch_public_project("owner/repo", REVISION, destination) == destination
    request, = observed
    assert request.full_url == f"https://github.com/owner/repo/archive/{REVISION}.zip"
    assert request.get_header("Authorization") is None
    assert (destination / "src/module.py").read_text() == "VALUE = 1\n"


def test_unavailable_public_revision_has_actionable_error(tmp_path, monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise bootstrap.urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)

    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(RuntimeError, match="Check that the repository is public") as error:
        bootstrap.fetch_public_project("owner/repo", REVISION, tmp_path / "source")
    assert error.value.__suppress_context__
    assert not (tmp_path / "source").exists()


def test_github_archive_redirect():
    request = urllib.request.Request(f"https://github.com/owner/repo/archive/{REVISION}.zip")
    url = f"https://codeload.github.com/owner/repo/zip/{REVISION}"
    redirected = bootstrap._GitHubRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, url)
    assert redirected.full_url == url


@pytest.mark.parametrize("url", ["http://codeload.github.com/a", "https://evil.example/a",
                                  "https://codeload.github.com.evil.example/a", "https://user@codeload.github.com/a"])
def test_unexpected_redirect_rejected(url):
    request = urllib.request.Request("https://github.com/a")
    with pytest.raises(RuntimeError):
        bootstrap._GitHubRedirectHandler().redirect_request(request, None, 302, "Found", {}, url)


@pytest.mark.parametrize("repo,revision", [("owner/repo?token=bad", REVISION), ("owner/repo", "main"),
                                           ("../repo", REVISION), ("owner/..", REVISION),
                                           ("owner/repo", "a" * 39)])
def test_repository_and_revision_validation(tmp_path, repo, revision):
    with pytest.raises(ValueError):
        bootstrap.fetch_public_project(repo, revision, tmp_path / "source")


@pytest.mark.parametrize("entries", [
    [("root/../outside", "bad")], [("/root/file", "bad")], [("root\\outside", "bad")],
    [("root/file", "one"), ("other/file", "two")],
    [("root/file", "one"), ("root/file", "two")],
    [("root/file", "one"), ("root/file/nested", "two")],
])
def test_unsafe_archive_is_transactional(tmp_path, entries):
    with pytest.raises(ValueError):
        bootstrap._extract_project_archive(archive_bytes(entries), tmp_path / "source")
    assert not (tmp_path / "source").exists()
    assert not (tmp_path / "outside").exists()


def test_symlink_archive_rejected(tmp_path):
    info = zipfile.ZipInfo("root/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ValueError):
        bootstrap._extract_project_archive(archive_bytes([(info, "../../outside")]), tmp_path / "source")


def test_existing_destination_is_preserved(tmp_path):
    destination = tmp_path / "source"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    with pytest.raises(ValueError):
        bootstrap.fetch_public_project("owner/repo", REVISION, destination)
    assert (destination / "keep").read_text() == "keep"


def test_oversized_download_is_rejected(tmp_path, monkeypatch):
    class Opener:
        def open(self, request, timeout):
            return io.BytesIO(b"x" * 11)
    monkeypatch.setattr(bootstrap, "_ARCHIVE_LIMIT", 10)
    monkeypatch.setattr(bootstrap.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(RuntimeError, match="exceeds the size limit"):
        bootstrap.fetch_public_project("owner/repo", REVISION, tmp_path / "source")


def test_oversized_extraction_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "_EXTRACTED_LIMIT", 3)
    with pytest.raises(ValueError, match="too large"):
        bootstrap._extract_project_archive(archive_bytes([("root/file", "long")]), tmp_path / "source")
