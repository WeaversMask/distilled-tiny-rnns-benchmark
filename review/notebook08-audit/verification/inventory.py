"""Snapshot the three input trees and file hashes without following symlinks."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import os
import stat

NAMES = {
    'new': 'main-study-rarity-v1',
    'baseline': 'main-study-first-tranche-v1',
    'smoke': 'rarity-smoke-a4582314a2b3',
}


def snapshot(root):
    root = root.absolute()
    resolved = root.resolve(strict=True)
    if resolved != root or not root.is_dir():
        raise ValueError(f'Input is not a canonical directory: {root}')
    files, directories, unsafe_entries = {}, [], []
    for path, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            child = Path(path) / name
            key = child.relative_to(root).as_posix()
            mode = child.lstat().st_mode
            if stat.S_ISREG(mode):
                digest = sha256()
                with child.open('rb') as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b''):
                        digest.update(block)
                files[key] = digest.hexdigest()
            elif stat.S_ISDIR(mode):
                directories.append(key)
            else:
                unsafe_entries.append(key)
    digest = sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'file_count': len(files), 'directory_count': len(directories),
            'file_map_sha256': digest, 'directories': sorted(directories), 'unsafe_entries': sorted(unsafe_entries),
            'file_sha256': dict(sorted(files.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (*NAMES, 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    parser.add_argument('--compare-to', type=Path, help='Require equality to an existing inventory')
    args = parser.parse_args()
    roots = {name: getattr(args, field) for field, name in NAMES.items()}
    output = args.output.resolve()
    if any(output.is_relative_to(root.resolve()) for root in roots.values()):
        parser.error('Output must be outside the three input trees')
    if args.compare_to and output == args.compare_to.resolve():
        parser.error('Output must differ from the comparison inventory')
    results = {name: snapshot(root) for name, root in roots.items()}
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    for name, item in results.items():
        print(name, item['file_count'], item['directory_count'], item['file_map_sha256'], item['unsafe_entries'])
    if args.compare_to:
        before = json.loads(args.compare_to.read_text())
        if before != results:
            raise SystemExit('Input inventories differ')
        print('Input inventories are identical')


if __name__ == '__main__':
    main()
