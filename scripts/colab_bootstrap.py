"""Download one pinned public GitHub revision for a Colab launcher.

This standard-library module is embedded into the launchers. Each invocation
installs into a fresh runtime-local destination; saved runs belong elsewhere.
"""
from pathlib import Path, PurePosixPath
import io
import os
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile


_ARCHIVE_LIMIT = 50 * 1024 * 1024
_EXTRACTED_LIMIT = 100 * 1024 * 1024


class _GitHubRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if (parsed.scheme != "https" or parsed.hostname != "codeload.github.com"
                or parsed.username is not None or parsed.password is not None
                or parsed.port not in (None, 443)):
            raise RuntimeError("GitHub returned an unexpected download destination")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _extract_project_archive(payload, destination):
    """Validate all archive paths before extracting a single GitHub root."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise ValueError("Use a fresh empty destination path for the project")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > 10000:
            raise ValueError("Invalid project archive entry count")
        roots, seen, planned = set(), set(), []
        total_size = 0
        for entry in entries:
            name = entry.filename
            components = name.rstrip("/").split("/")
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if (name.startswith("/") or "\\" in name or any(p in ("", ".", "..") for p in components)
                    or mode not in (0, stat.S_IFREG, stat.S_IFDIR) or entry.flag_bits & 1):
                raise ValueError("Unsafe project archive entry")
            roots.add(components[0])
            relative = PurePosixPath(*components[1:])
            normalized = str(relative)
            if normalized in seen:
                raise ValueError("Duplicate project archive entry")
            seen.add(normalized)
            if len(components) == 1:
                if not entry.is_dir():
                    raise ValueError("Project archive must contain a root directory")
                continue
            total_size += entry.file_size
            if total_size > _EXTRACTED_LIMIT:
                raise ValueError("Project archive is too large")
            planned.append((entry, relative))
        if len(roots) != 1 or not planned:
            raise ValueError("Project archive must have one root directory")
        files = {str(path) for entry, path in planned if not entry.is_dir()}
        for _, path in planned:
            if any(str(parent) in files for parent in path.parents):
                raise ValueError("Conflicting project archive paths")
        staging = Path(tempfile.mkdtemp(prefix=".colab-project-", dir=destination.parent))
        try:
            for entry, relative in planned:
                target = staging.joinpath(*relative.parts)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            # Caller selects an unused path. Recheck to avoid replacing a path
            # that appeared while downloading/extracting.
            if os.path.lexists(destination):
                raise ValueError("Project destination already exists")
            staging.rename(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return destination


def fetch_public_project(repo, commit, destination):
    """Fetch and safely extract a public ZIP pinned to a full Git commit."""
    if (not isinstance(repo, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}", repo)
            or repo.rsplit("/", 1)[-1] in {".", ".."}):
        raise ValueError("Repository must be an owner/name pair")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("A full 40-character Git commit is required")
    if os.path.lexists(destination):
        raise ValueError("Use a fresh destination path for the project")
    request = urllib.request.Request(
        f"https://github.com/{repo}/archive/{commit}.zip",
        headers={"User-Agent": "distilled-rnns-colab"},
    )
    try:
        opener = urllib.request.build_opener(_GitHubRedirectHandler())
        with opener.open(request, timeout=60) as response:
            payload = response.read(_ARCHIVE_LIMIT + 1)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise RuntimeError(
                "GitHub could not find the pinned revision. Check that the repository is public."
            ) from None
        raise RuntimeError(f"GitHub project download failed (HTTP {error.code}).") from None
    except (OSError, RuntimeError, ValueError):
        raise RuntimeError("GitHub project download failed. Check your connection and try again.") from None
    if len(payload) > _ARCHIVE_LIMIT:
        raise RuntimeError("The project download exceeds the size limit")
    try:
        return _extract_project_archive(payload, destination)
    except Exception:
        raise RuntimeError("The downloaded project archive could not be safely installed") from None
