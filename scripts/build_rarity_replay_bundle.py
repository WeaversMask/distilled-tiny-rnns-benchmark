"""Package the audited Notebook 08 inputs needed to replay the primary results."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


RUN_NAME = "main-study-rarity-v1"
REPLAY_FILES = {"data.npz", "predictions.npz", "probe_predictions.npz", "report.json"}
INVENTORY = Path(__file__).resolve().parents[1] / "review/notebook08-audit/verification/inventory-before.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle(run, output):
    run = run.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(run):
        raise ValueError("the output must be outside the audited run folder")

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))[RUN_NAME]["file_sha256"]
    populations = {}
    for name, checksum in inventory.items():
        path = PurePosixPath(name)
        if len(path.parts) == 5 and path.parts[0] == "cells" and path.name in REPLAY_FILES:
            populations.setdefault(path.parent, {})[path.name] = checksum
    if len(populations) != 18 or any(files.keys() != REPLAY_FILES for files in populations.values()):
        raise ValueError("the audited inventory must contain all four replay files for 18 populations")

    with TemporaryDirectory(prefix="rarity-replay-", dir=output.parent) as staging:
        staged = Path(staging) / output.name
        with ZipFile(staged, "w") as archive:
            for population, files in sorted(populations.items()):
                for name, expected_sha256 in sorted(files.items()):
                    relative = population / name
                    path = run / relative
                    if path.resolve() != path:
                        raise ValueError(f"expected an original file, found a symlink: {relative}")
                    payload = path.read_bytes()
                    if hashlib.sha256(payload).hexdigest() != expected_sha256:
                        raise ValueError(f"audited SHA-256 mismatch: {relative}")
                    member = ZipInfo((PurePosixPath(RUN_NAME) / relative).as_posix())
                    member.create_system = 3
                    member.external_attr = 0o100644 << 16
                    archive.writestr(member, payload, compress_type=ZIP_DEFLATED, compresslevel=6)
        checksum = file_sha256(staged)
        staged.replace(output)
    return checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rarity-run", type=Path, required=True, help="Audited Notebook 08 result folder")
    parser.add_argument("--output", type=Path, required=True, help="Output ZIP file")
    args = parser.parse_args()
    try:
        checksum = build_bundle(args.rarity_run, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{checksum}  {args.output.name}")


if __name__ == "__main__":
    main()
