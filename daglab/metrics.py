import numpy as np

def paired(rows,baseline,bootstrap=2000):
    gains=np.array([100*(r[baseline]-r['student'])/r[baseline] for r in rows])
    groups={}
    for r,g in zip(rows,gains):groups.setdefault(r['latent_id'],[]).append(float(g))
    values=list(groups.values());rng=np.random.default_rng(271828);boots=[]
    if len(values)>1:
        sums=np.array([sum(x) for x in values]);counts=np.array([len(x) for x in values])
        for _ in range(bootstrap):
            ix=rng.integers(len(values),size=len(values));boots.append(float(sums[ix].sum()/counts[ix].sum()))
    return {'baseline':baseline,'configs':len(rows),'latent_graphs':len(values),'mean_paired_gain_percent':float(gains.mean()),'wins':int((gains>1e-9).sum()),'ties':int((np.abs(gains)<=1e-9).sum()),'losses':int((gains < -1e-9).sum()),'graph_cluster_CI95_percent':np.quantile(boots,[.025,.975]).tolist() if boots else None}
