"""Download fixed author revisions only; do not execute remote source."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/faithful_external_20261102/sources'
ITEMS = [('cai', '9erAlpenjaeger/DAG-CAI-DR', '1a2aacc3fc44d8d8f1ef779488a39cab80abf991'),
         ('giph', 'uidmice/placement-rl', '427caced16988f0f7c22b7dcceea59fa472dbc34')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for name, repo, commit in ITEMS:
        url = f'https://codeload.github.com/{repo}/zip/{commit}'
        target = OUT / f'{name}.zip'
        if not target.exists():
            blob = urllib.request.urlopen(url, timeout=60).read()
            with zipfile.ZipFile(__import__('io').BytesIO(blob)) as archive:
                assert archive.testzip() is None
            target.write_bytes(blob)
        with zipfile.ZipFile(target) as archive:
            for info in archive.infolist():
                relative = Path(*Path(info.filename).parts[1:])
                if '..' in relative.parts or relative.is_absolute():
                    raise ValueError(info.filename)
                destination = OUT / name / relative
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(info))
        entry = dict(name=name, repo=repo, commit=commit, url=url,
                     sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size)
        records.append(entry)
        print(json.dumps(entry), flush=True)
    (OUT / 'MANIFEST.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
