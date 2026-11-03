#!/usr/bin/env python3
"""Rebuild small runtime SQLite caches from the compact, verified logical data."""
import argparse,gzip,hashlib,json,sqlite3,sys,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from daglab.sft_three_panel import pack

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    manifest=json.loads((ROOT/'data/MANIFEST.json').read_text())
    for shard in manifest['train_shards']+list(manifest['panels'].values()):
        assert sha(ROOT/shard['path'])==shard['sha256'],shard['path']
    marker=ROOT/'data/runtime/PREPARED.json'
    if marker.exists():
        assert json.loads(marker.read_text())['packed_manifest_sha256']==sha(ROOT/'data/MANIFEST.json')
        print('Already prepared; verified all packed shard hashes.');return
    cfg=json.loads((ROOT/'configs/teacher_search_ablation_ppo_20261102.json').read_text())
    roots={v:ROOT/cfg[k] for v,k in [('legacy','original_cache'),('strong','strong_teacher_cache'),('no_search','no_search_cache')]}
    connections={};physical=hashlib.sha256();bases=set();count=0
    for variant,path in roots.items():
        path.mkdir(parents=True,exist_ok=True);target=path/'public.sqlite3'
        if target.exists():raise RuntimeError('Incomplete prior prepare: remove only generated *.sqlite3 before retrying')
        con=sqlite3.connect(target);con.execute('PRAGMA synchronous=OFF')
        con.execute('CREATE TABLE records(key TEXT PRIMARY KEY,source TEXT,role TEXT,cohort INTEGER,base_id TEXT,topology TEXT,n INTEGER,k INTEGER,physical_sha TEXT,raw BLOB,payload BLOB)')
        connections[variant]=con
    for shard in manifest['train_shards']:
        with gzip.open(ROOT/shard['path'],'rt') as stream:
            for line in stream:
                item=json.loads(line);columns=item['columns']; count+=1;bases.add(columns[4])
                physical.update((columns[0]+' '+columns[8]+'\n').encode())
                for variant,con in connections.items():
                    label=item['teachers'][variant];raw=dict(item['raw'],quality=label['quality'],supervision=label['supervision'])
                    payload=dict(item['payload'],supervision=label['supervision'],teacher=label['teacher'])
                    for field in ['teacher_version','legacy_teacher_value']:
                        if field in label:payload[field]=label[field]
                    if 'teacher_version' in label:raw['teacher_version']=label['teacher_version']
                    if 'payload_meta' in label:payload['meta']=label['payload_meta']
                    con.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?)',columns+[pack(raw),pack(payload)])
        for con in connections.values():con.commit()
        print('PREPARE',count,'/',manifest['train_configs'],flush=True)
    assert (count,len(bases))==(17261,10000)
    assert physical.hexdigest()==manifest['physical_membership_sha256']
    for variant,con in connections.items():
        con.execute('CREATE INDEX source_role ON records(source,role)');con.commit();con.close()
        path=roots[variant]/'MANIFEST.json'
        if path.exists():
            original=ROOT/'evidence/original_manifests'/variant/'MANIFEST.json';original.parent.mkdir(parents=True,exist_ok=True)
            if not original.exists():original.write_bytes(path.read_bytes())
        path.write_text(json.dumps(dict(status='complete',portable_logical_snapshot=True,configs=17261,bases=10000,
            graph_and_action_caches='generated on demand',database_sha256=sha(roots[variant]/'public.sqlite3'),
            packed_manifest_sha256=sha(ROOT/'data/MANIFEST.json')),indent=2)+'\n')
    # Keep WF960 in the original cache too, as historical baseline helpers expect.
    con=sqlite3.connect(roots['legacy']/'public.sqlite3')
    with gzip.open(ROOT/manifest['panels']['wf960']['path'],'rt') as stream:
        for line in stream:
            row=json.loads(line);p=row['payload'];m=p['meta'];raw=row['raw']
            con.execute('INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?)',[row['key'],'wfinstances','eval',m.get('cohort',0),
                p['base_id'],m.get('topology',''),p['n'],raw['input']['k'] if 'input' in raw else raw.get('k',0),
                '',pack(raw),pack(p)])
    con.commit();con.close()
    original_manifest=roots['legacy']/'MANIFEST.json';obj=json.loads(original_manifest.read_text());obj['database_sha256']=sha(roots['legacy']/'public.sqlite3');original_manifest.write_text(json.dumps(obj,indent=2)+'\n')
    def create_validation(path,names):
        path.parent.mkdir(parents=True,exist_ok=True)
        con=sqlite3.connect(path);con.execute('CREATE TABLE records(key TEXT PRIMARY KEY,panel TEXT,raw BLOB,payload BLOB)')
        for name in names:
            panel='parrot_pattern_templates' if name=='parrot498' else 'wf_by_application' if name.startswith('wf') else name
            with gzip.open(ROOT/manifest['panels'][name]['path'],'rt') as stream:
                for line in stream:
                    row=json.loads(line);con.execute('INSERT INTO records VALUES(?,?,?,?)',(row['key'],panel,pack(row['raw']),pack(row['payload'])))
        con.commit();con.close()
    create_validation(ROOT/cfg['validation_cache'],['synthetic_val862','parrot498'])
    for panel in ['wf960','wf400']:create_validation(ROOT/f'data/runtime/{panel}.sqlite3',[panel])
    marker.parent.mkdir(parents=True,exist_ok=True)
    marker.write_text(json.dumps(dict(status='complete',train=17261,bases=10000,panels={k:v['records'] for k,v in manifest['panels'].items()},
        packed_manifest_sha256=sha(ROOT/'data/MANIFEST.json')),indent=2)+'\n')
    print('READY: 17261 TRAIN, 862 VAL, WF960/WF400 and constructed Parrot498. Features/traces are lazy caches.')

if __name__=='__main__':main()
