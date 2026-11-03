# Handoff scope

This is a compact transfer snapshot, not the entire historical research tree.
Read README.md, data/MANIFEST.json, evidence/BASELINE_SNAPSHOT.json and
evidence/PORTABILITY_CHANGES.json before starting work. No job is running merely
because a copied historical progress file says "running"; no host PID/lock files
are deployment controls. Mac original experiments continue separately.

TRAIN17261/10000graphs, synthetic VAL862/500graphs; WF960 full historical panel
and WF400 focused panel are separate memberships, plus accurately named
constructed Parrot498. Keep application groups separate. These are known
development panels. No TEST is shipped. Preserve negative results and physically
validate schedules. Use tools/prepare.py then tools/verify.py before new training.

Use tools/train_ppo.py for fresh PPO from either supplied SFT model; fresh output
directories preserve historical checkpoints. Same32612-parameter StateActor,
one full actor forward then fixedHEFT physical construction; critic training
does not add deployment actor calls. Record every config/data/source change.

GiPH paper200 and CAI snapshots are unfinished unless a complete final artifact
says otherwise. tools/resume_baseline.py records explicit portability binding
migration while retaining model/Adam/RNG state; it never silently resumes the
canceled full17261-episode GiPH plan. GOODRL is pinned author source in its
original scenario, not an already adapted baseline; WeCAN author code was not
verified publicly available. Do not claim either has measured results here.
