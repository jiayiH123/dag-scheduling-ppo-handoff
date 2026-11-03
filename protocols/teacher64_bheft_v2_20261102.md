# Teacher64+BHEFT-v2 with legacy-result retention

2026-11-02. User explicitly requests starting a new Teacher64 version adding BHEFT as candidate. Only label generation is launched; no model training, order change, VAL member change or automatic expansion.

## Fixed inputs and scientific comparison

Versioned output `runs/teacher64_bheft_v2_20261102`, source caches opened read-only. First process original17261 configurations, then current4000 augmentation, retaining source membership separately. Total21261 configurations,1589632 actions, same physical inputs/features/budget and fixed HEFT construction order. Original-only subset remains directly selectable for a teacher-only comparison. Do not consume any VAL or TEST labels as TRAIN, or replace historical Teacher64 evaluation lines.

## New teacher definition

The conceptual initial pool is the original ten rules plus BHEFT-adapted, evaluated under the same fixed HEFT order. If the original ten fixed-order (C,cost) values and initial rule index are already stored, verify their exact selection and reconstruct that best rule's mapping; physically replay and require exact agreement. This is equivalent to recomputing the ten-rule minimum, while avoiding redundant work. Where cached per-rule provenance is unavailable, execute all ten rules and fixed-order replays. Keep all ten initial values in provenance. BHEFT must pass identical order and budget/resource checks.

Select the minimum(C,cost) of the old ten-rule best and BHEFT; ties retain the old first-winner convention. Use the same instance seed+3917 random stream and exactly64 one/two-task proposals as original Teacher64, accepting only lexicographically lower(C,cost). Then compare the resulting final mapping with the stored old Teacher64 FINAL mapping, retaining whichever has lower(C,cost); prefer the old mapping on exact equality to avoid needless label changes. The old final mapping is a retention safeguard after new search, not the new search's start. This isolates adding BHEFT to the original starting pool while preventing an alternate local-search path from degrading old labels.

The name means64 new proposals per instance; legacy offline64-proposal work is reused and disclosed, not counted as zero total historical search. Record attempted, feasible, accepted and actual fresh simulations/runtime. This is not an optimality certificate.

## Cache and validation gates

Write a NEW standalone `public.sqlite3` with the same training-record schema and exact source keys. Raw physical fields and graph tensors are unchanged; new supervised mappings, makespan/cost, starts/ends and StateScore decisions become authoritative in the new raw/payload. Store legacy objective, input/graph hashes and per-case teacher provenance. Regenerate decisions on each new mapping's OWN fixed-order trajectory, never reuse changed mappings' old masks/offsets/gap/wait/prefix state.

Before full build, engineer representative original n20/40/80/120×k3/4/6, all retained target-app augmentation strata and largest n280×k6 where present. Verify original objective replay, cached-seed reconstruction, BHEFT schedule replay, exact old-compatible RNG behavior when the new start is unchanged, mapping eligibility and hard budget, new objective no worse than old-final or BHEFT, fresh StateScore physical replay and unchanged graph tensors. Compare serial and worker results on an engineering case. Engineering samples are correctness gates, not quality estimates.

Full build is a finite resumable two-worker process with exclusive lock, atomic batch commits, code/config/input binding, crash evidence, progress heartbeat and job logs; no unbounded experiment loop. All cases independently replayed/validated before insertion. On completion audit exact21261 keys,source counts/action counts,per-case dominance and database integrity. Save per-source and family/n/k/target-app gain coverage, magnitude, final incumbent source, label/action changes and cost-only changes. Report full original17261 quality results before drawing conclusions about improved student performance.

## BUDA question

Separately compare the original IWQoS2022 paper with local adaptation, distinguish modeled target differences from demonstrated implementation sensitivity and untested hypotheses. Existing unit-balanced results are diagnostics, not the native method. No BUDA algorithm tuning or replacement is launched. Any small unit-invariance probe is engineering evidence of representational sensitivity, not a benchmark quality estimate.
