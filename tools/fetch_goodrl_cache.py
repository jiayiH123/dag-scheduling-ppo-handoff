#!/usr/bin/env python3
"""Optional upstream-generated HEFT memory files omitted from the small bundle."""
import json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'third_party/GOODRL_UPSTREAM.json').read_text())
for row in manifest['excluded_large_cache']:
    path=ROOT/'third_party/GOODRL'/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==row['bytes']:continue
    temp=path.with_suffix(path.suffix+'.download')
    with urllib.request.urlopen(row['url'],timeout=180) as source,temp.open('wb') as target:
        while True:
            chunk=source.read(1024*1024)
            if not chunk:break
            target.write(chunk)
    assert temp.stat().st_size==row['bytes'];temp.replace(path);print(row['path'])
