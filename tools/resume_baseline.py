#!/usr/bin/env python3
"""Declared relocation of baseline checkpoint bindings, then original workers."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=['cai','giph'],required=True)
    parser.add_argument('--seed',type=int,choices=[11,22,33],default=11)
    parser.add_argument('--migrate-only',action='store_true');args=parser.parse_args()
    import daglab.faithful_external as f
    from daglab.faithful_external_data import Inputs
    from scripts.train_faithful_external_20261102 import binding
    from scripts.run_giph_paper200_20261102 import bound
    from daglab.structured_dataset import sha256
    from scripts.train_faithful_external_20261102 import save_checkpoint
    f.torch.set_num_threads(1)
    manifest=json.loads((ROOT/'data/runtime/PREPARED.json').read_text());assert manifest['status']=='complete'
    # Reconstructing SQLite changes file bytes, never physical inputs; use the
    # independently verified packed manifest and preserve every old binding.
    inputs=Inputs();inputs.audit();cai_binding=binding(inputs);giph_binding=bound(inputs)
    migration=ROOT/'evidence/portable_baseline_migration';migration.mkdir(exist_ok=True)
    roots=[(f.RUN,cai_binding,'cai_s*'),(ROOT/'runs/giph_paper200_20261102',giph_binding,'seed*')]
    for base,new_binding,pattern in roots:
        for directory in base.glob(pattern):
            if not directory.is_dir():continue
            for checkpoint in directory.glob('*.pt'):
                saved=f.torch.load(checkpoint,map_location='cpu',weights_only=False)
                if saved.get('binding')==new_binding:continue
                old=saved.get('binding');old_sha=sha256(checkpoint)
                saved['binding']=new_binding;save_checkpoint(checkpoint,saved)
                target=migration/(directory.name+'_'+checkpoint.name+'.json')
                if not target.exists():target.write_text(json.dumps(dict(checkpoint=str(checkpoint.relative_to(ROOT)),
                    before_sha256=old_sha,after_sha256=sha256(checkpoint),old_binding=old,new_binding=new_binding,
                    reason='Explicit portable relocation: equivalent physical records, regenerated SQLite bytes and interpreter path adapter; model/Adam/RNG/counters unchanged'),indent=2)+'\n')
            for name in ['COMPLETE.json','TRAINING_COMPLETE.json','metadata.json']:
                path=directory/name
                if path.exists():
                    obj=json.loads(path.read_text())
                    if 'binding' in obj:obj['binding']=new_binding
                    path.write_text(json.dumps(obj,indent=2)+'\n')
    checks=ROOT/'runs/giph_paper200_20261102/ENGINEERING_CHECKS.json'
    check=json.loads(checks.read_text());check['binding']=giph_binding;checks.write_text(json.dumps(check,indent=2)+'\n')
    print('Portable binding migration recorded; model, optimizer, RNG and episode/update counters retained.',flush=True)
    if args.migrate_only:return
    if args.method=='giph':
        from scripts.run_giph_paper200_20261102 import worker
        worker(args.seed)
    else:
        from scripts.run_faithful_external_20261102 import method_queue
        method_queue('cai')

if __name__=='__main__':main()
