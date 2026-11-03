"""Build a compact logical snapshot; never copy host dependencies or credentials."""
import gzip, hashlib, json, shutil, sqlite3, sys, time, zlib
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
SOURCE = OUT.parent / 'dag-scheduling-llm-lab'

def unpack(blob): return json.loads(zlib.decompress(blob))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def copy(relative):
    source = SOURCE / relative
    if not source.is_file(): return
    target = OUT / relative; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

def export():
    paths = json.loads(Path('/private/tmp/ppo_handoff_code_paths.json').read_text())
    for path in paths: copy(path)
    for path in SOURCE.glob('vendor/external_baseline_sources_20261028/**/*'):
        if path.is_file() and '__pycache__' not in path.parts: copy(str(path.relative_to(SOURCE)))
    for pattern in ['configs/*.json', 'protocols/*20261102.md']:
        for path in SOURCE.glob(pattern): copy(str(path.relative_to(SOURCE)))
    # Historical, unedited protocol/config copies; active portable config is separate.
    for path in (OUT/'configs').glob('*.json'):
        if path.name not in ['teacher_search_ablation_ppo_20261102.json','faithful_external_20261102.json','giph_paper200_20261102.json',
            'sft_teacher64_bheft_original_20261102.json','sft_portfolio11_no_search_20261102.json','sft_three_panel_20261101.json',
            'sft_three_panel_run_20261101.json','teacher64_bheft_v2_20261102.json','external_baselines_20261028.json']:
            path.unlink()
    # Only Python/text author modules needed by the imported core; no .so/.dylib.
    for method in ['cai','giph']:
        base = SOURCE/'runs/faithful_external_20261102/sources'/method
        for path in base.rglob('*'):
            if path.is_file() and ('__pycache__' not in path.parts) and (path.suffix in ['.py','.md','.txt','.yaml','.yml','.json'] or 'license' in path.name.lower()):
                copy(str(path.relative_to(SOURCE)))
    for relative in ['runs/faithful_external_20261102/sources/MANIFEST.json',
                     'runs/faithful_external_20261102/INPUT_AUDIT.json',
                     'runs/faithful_external_20261102/GIPH_HEAVY_BUDGET_STOP.json',
                     'runs/faithful_external_20261102/GIPH_CANCELED_WORK_ACCOUNTING.json']:
        copy(relative)
    for path in (SOURCE/'runs/faithful_external_20261102/engineering').glob('*.json'): copy(str(path.relative_to(SOURCE)))
    configs = json.loads((OUT/'configs/teacher_search_ablation_ppo_20261102.json').read_text())
    sft_roots = [configs['strong_SFT_run'], configs['no_search_SFT_run']]
    for root in sft_roots:
        for name in ['best.pt','last.pt','metadata.json','summary.json','config.json','training.jsonl',
                     'val_synthetic_val862_curve.json','INPUT_AUDIT.json','ENGINEERING_CHECKS.json']:
            copy(str(Path(root)/name))
        for path in (SOURCE/root).glob('val_synthetic_val862_*.json.gz'): copy(str(path.relative_to(SOURCE)))
    small_roots = ['runs/sft_teacher64_bheft_original_20261102/new_teacher_val862',
                   'runs/teacher_search_ablation_ppo_20261102/no_search_teacher_val862',
                   'runs/teacher_search_ablation_ppo_20261102/final_analysis',
                   'runs/sft_teacher64_bheft_original_20261102/all_baselines_val862']
    for root in small_roots:
        for path in (SOURCE/root).glob('*'):
            if path.is_file() and path.suffix in ['.json','.csv','.md','.png','.pdf']: copy(str(path.relative_to(SOURCE)))
    external=json.loads((SOURCE/'runs/teacher_search_ablation_ppo_20261102/final_analysis/EXTERNAL_BASELINES_VAL862.json').read_text())
    for artifact in external['artifacts']: copy(artifact['path'])
    for root in [configs['original_cache'], configs['strong_teacher_cache'], configs['no_search_cache'],'runs/sft_three_panel_20261101/data']:
        for name in ['MANIFEST.json','FULL_AUDIT.json','TRAIN_PROBE.json']: copy(str(Path(root)/name))
    for relative in ['runs/original_teacher_sources_20261102/PER_CONFIGURATION.jsonl',
                     'runs/synthetic_transfer_20261031/S0_original/metadata.json',
                     'runs/sft_three_panel_scope_20261101/DATA_SCOPE_LOCK.json',
                     'runs/sft_three_panel_scope_20261101/VERIFICATION.json',
                     'data/wf_application_panels_20261101/REGISTRY.json',
                     'data/wf_application_panels_20261101/README.md']:
        copy(relative)
    for name in ['problem_model.pdf','TEACHER_SEARCH_ABLATION_PPO_RESULTS_20261102.md',
                 'TEACHER64_BHEFT_V2_20261102.md','BUDA_SCENARIO_AND_FAILURE_AUDIT_20261102.md',
                 'EXTERNAL_BASELINES_CURRENT_VAL862_20261102.md','FAITHFUL_EXTERNAL_REPRODUCTION_20261102.md',
                 'SYNTHETIC_TRAIN_VAL_DISTRIBUTION_20261102.md','GOODRL_CRITIC_TRANSFER_PLAN_20261027.md',
                 'GOODRL_CRITIC_RESULTS_20261027.md','wecan2026.md','WF_GROUP_EVALUATION_20261101.md']:
        copy('docs/'+name)
    for pattern in ['runs/teacher_search_ablation_ppo_20261102/ppo_*/curve.json',
                    'runs/teacher_search_ablation_ppo_20261102/ppo_*/summary.json',
                    'runs/teacher_search_ablation_ppo_20261102/ppo_*/metadata.json']:
        for path in SOURCE.glob(pattern): copy(str(path.relative_to(SOURCE)))
    # Pack one graph/input and three teacher label sets, rather than three caches.
    def connect(path): return sqlite3.connect('file:'+str(SOURCE/path)+'?mode=ro',uri=True)
    base=connect(Path(configs['original_cache'])/'public.sqlite3')
    strong=connect(Path(configs['strong_teacher_cache'])/'public.sqlite3')
    nosearch=connect(Path(configs['no_search_cache'])/'public.sqlite3')
    columns='key,source,role,cohort,base_id,topology,n,k,physical_sha,raw,payload'
    query='SELECT '+columns+' FROM records WHERE source="synthetic" AND role="train" ORDER BY key'
    stream=None; shards=[]; count=0; bases=set(); actions=0; start=time.time()
    physical=hashlib.sha256()
    for a,b,c in zip(base.execute(query),strong.execute(query),nosearch.execute(query)):
        assert a[:9]==b[:9]==c[:9]
        if count%256==0:
            if stream: stream.close(); shards[-1].update(bytes=target.stat().st_size,sha256=sha(target))
            target=OUT/'data/packed'/f'train-{count//256:03d}.jsonl.gz'
            stream=gzip.open(target,'wt',compresslevel=6)
            shards.append(dict(path=str(target.relative_to(OUT)),records=0))
        raw=[unpack(row[9]) for row in [a,b,c]]; payload=[unpack(row[10]) for row in [a,b,c]]
        assert raw[0]['input']==raw[1]['input']==raw[2]['input']
        teachers={}
        for name,r,p in zip(['legacy','strong','no_search'],raw,payload):
            supervision=p['supervision']; assert supervision['decisions']==r['supervision']['decisions']
            teachers[name]=dict(supervision={k:v for k,v in supervision.items() if k!='decisions'},quality=r['quality'],teacher=p['teacher'])
            if p['meta']!=payload[0]['meta']: teachers[name]['payload_meta']=p['meta']
            for field in ['teacher_version','legacy_teacher_value']:
                if field in p: teachers[name][field]=p[field]
        raw0={k:v for k,v in raw[0].items() if k not in ['supervision','quality']}
        payload0={k:v for k,v in payload[0].items() if k not in ['supervision','decisions','teacher','graph']}
        record=dict(columns=list(a[:9]),raw=raw0,payload=payload0,teachers=teachers)
        stream.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
        physical.update((a[0]+' '+a[8]+'\n').encode()); bases.add(a[4]); actions+=a[6];count+=1;shards[-1]['records']+=1
        if count%256==0:
            print('TRAIN_PACK',count,'seconds',round(time.time()-start,1),flush=True)
    stream.close();shards[-1].update(bytes=target.stat().st_size,sha256=sha(target))
    assert (count,len(bases),actions)==(17261,10000,1122860)
    panels={}
    def panel(db,where,name,expected):
        con=connect(Path(db)); target=OUT/'data/packed'/f'{name}.jsonl.gz';n=0
        with gzip.open(target,'wt',compresslevel=6) as stream:
            for key,raw,payload in con.execute('SELECT key,raw,payload FROM records WHERE '+where+' ORDER BY key'):
                value=unpack(payload);value.pop('graph',None)
                stream.write(json.dumps(dict(key=key,raw=unpack(raw),payload=value),separators=(',',':'))+'\n');n+=1
        assert n==expected; panels[name]=dict(path=str(target.relative_to(OUT)),records=n,bytes=target.stat().st_size,sha256=sha(target));con.close()
    panel(configs['validation_cache'],'panel="synthetic_val862"','synthetic_val862',862)
    panel(Path(configs['original_cache'])/'public.sqlite3','source="wfinstances" AND role="eval"','wf960',960)
    panel('runs/application_augmentation_4000_20261102/data/validation.sqlite3','panel="wf_by_application"','wf400',400)
    # Small constructed panel was also used historically; preserve it with accurate naming.
    panel('runs/sft_three_panel_20261101/data/validation.sqlite3','panel="parrot_pattern_templates"','parrot498',498)
    manifest=dict(schema='portable-logical-cache-v1',train_configs=count,train_bases=len(bases),actions_per_epoch=actions,
        teacher_variants=['legacy','strong','no_search'],physical_membership_sha256=physical.hexdigest(),
        train_shards=shards,panels=panels,source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip(),
        raw_or_graph_duplication_across_teachers=False,graph_features='recomputed lazily with unchanged deterministic feature builder',
        teacher_prefix_decisions='replayed lazily from stored mappings with unchanged StateScoreDecoder; exact C/cost/start/end asserted',TEST_included=False)
    (OUT/'data/MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PACK_COMPLETE',count,'MB',round(sum(r['bytes'] for r in shards+list(panels.values()))/1e6,2),flush=True)

if __name__=='__main__':export()
