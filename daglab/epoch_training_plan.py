"""Pure, deterministic full-corpus epoch sampling and observation scheduling."""
import math
import random
from .offline_data import digest


def make_epoch_plan(con,sources,seed,epochs,batch_size):
    placeholders=','.join('?' for _ in sources)
    rows=list(con.execute(f'SELECT key,source FROM records WHERE role="train" AND source IN ({placeholders}) ORDER BY key',sources))
    keys=[r[0] for r in rows];source_for_key=dict(rows)
    assert keys and len(keys)==len(set(keys))
    batches=[]
    for epoch in range(1,epochs+1):
        shuffled=list(keys);random.Random(f'{seed}:configuration_epoch:{epoch}').shuffle(shuffled)
        for start in range(0,len(shuffled),batch_size):batches.append(shuffled[start:start+batch_size])
    return batches,source_for_key,digest(batches)


def observation_steps(updates_per_epoch,epochs,per_epoch=5):
    assert updates_per_epoch>0 and epochs>0 and per_epoch>0
    local=sorted({math.ceil(updates_per_epoch*j/per_epoch) for j in range(1,per_epoch+1)})
    return [0]+[(e*updates_per_epoch)+i for e in range(epochs) for i in local]
