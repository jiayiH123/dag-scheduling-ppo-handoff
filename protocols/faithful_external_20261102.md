# Paper-faithful CAI / GiPH under the shared scheduling physics

User explicitly requested faithful reproduction, tuning on TRAIN17261 / VAL862,
and final VAL reporting. This supersedes the earlier no-more-training restriction
for these two controls only. Known VAL is development/selection data, not an
unseen confirmation test. No public/WF/TPC inputs, SFT labels or TEST used.

## Sources and fidelity hierarchy

Pinned author revisions: CAI 1a2aacc3fc44d8d8f1ef779488a39cab80abf991;
GiPH 427caced16988f0f7c22b7dcceea59fa472dbc34. Full archives in
`runs/faithful_external_20261102/sources`, archive hashes in its MANIFEST.
Do not modify the source archives or the old experiments. When release and paper
conflict, use the explicitly stated main-paper network/hyperparameters and
document the conflict; numerical reproduction of published tables is not claimed.

CAI Appendix C.2: Graphormer 4 layers / 4 heads / 64 embedding, 64-64-1 ReLU
head, Adam5e-4, batch16, logit regularization .001. Release Graphormer call
expects 3 return values but its encoder returns 2; caller supplies continuous
PyG Data although the included encoder expects a categorical dictionary. Its
BiDir8+8/8head settings also differ from the paper. Reuse the actual published
Graphormer encoder/layers/attention; provide numeric node inputs and dictionary
graph preprocessing, using the paper dimensions. Fairseq zero-noise/dropout/
LayerNorm wrappers may be replaced by their Torch equivalents; test attention
forward/gradient equivalence. No replacement Transformer architecture.
CAI sampling/probabilities: actual author GumbelTopoSortAdj, correctly transposed
adjacency (rows=child, columns=parent), one static Gumbel draw per node, ready-set
likelihood indexed by task ID. Dense advantage follows released HEFT individual
finish-time difference, no per-instance C normalization. Preserve nonzero global
batch normalization (unbiased std), clipping[-2,2], selected-logprob entropy-like
term .01, gradient norm .5; fixed padding120. Guard empty/one nonzero advantages
to remain finite. Four sampled orders per graph as a recorded training-budget
choice, not a paper-provided default. Greedy unperturbed evaluation, no search.

GiPH: execute actual author FNN, GiPHEmbedding, MPLayer, SoftmaxActor with DGL;
author native defaults embedding10 / head hidden16, and default Adam .01,
gamma .97, online one-episode optimizer update. Categorical sampling in both
training and evaluation, 2n moves in both, original running-prefix reward
baseline and discounted returns, no added entropy/BC/PPO or gradient clipping.
Reward is raw Cprev-Cnext as in author code, no normalization by initial C.
Keep best visited including initial, and all actor-call/search accounting.
Restore [-1,1] minmax features, data-weighted initial criticalities and author
Program.update_criticality(.1). History resets at each independent VAL case/repeat;
training history may persist per TRAIN config and is checkpointed.

## Necessary physical adaptations (explicitly retained)

Use existing exact nonpreemptive cumulative SGS and directed fixed overlapping
communication; monetary hard budget. CAI EFT uses minimum-cost suffix reserve.
GiPH masks ineligible, overbudget, no-change and consecutive-same-task actions;
if no valid move stop rather than violate a hard constraint. Initial random maps
are sampled feasible using the same suffix reserve. HEFT-start GiPH is reported
as an additional warm-start variant, never confused with original random start.
Current HEFT construction order is only the deterministic mapping evaluator for
GiPH; CAI learns its own dependency-valid order.

Append physical observables needed for budget/capacity decisions to original
features; do not erase global budget/platform attributes by graph centering.
GiPH compute/comm features use actual durations/delays from our simulator.
Its ID0 start-potential special case is generalized to actual DAG roots because
our node IDs are permuted and multiple roots are allowed. Constant minmax columns
become0, including edge features; zero-byte criticality initialization is uniform
(finite equal-data limit). Criticality follows precedence-chain timing under
the common resource-aware simulator. These repairs are not claimed original
data conditions. Independently compare features against author methods with only
these stated finite/root guards; all final schedules physically validated.

## Finite tuning / training plan

Do engineering and throughput measurement first; engineering states never become
formal checkpoints. All runs use original full17261 inputs, identical membership
and seeded shuffle, no cherry-picked small-graph training.

CAI seed11 rates [5e-4 (paper),1e-4,1e-3], each7 complete epochs; select one
recipe/checkpoint by full VAL base-macro C/HEFT. Run selected recipe seeds22/33
for7epochs. Evaluate at0, each500 updates and epoch ends, retain all values.
This is5 finite runs, no open-ended hyperparameter expansion.

GiPH seed11 rates [.01 (paper),.003,.001], each one complete17261-config pass
with2n moves and online updates. Select using random-start full-VAL base-macro
C/HEFT, with fixed per-case action seeds (not best-of-seeds). Run selected rate
seeds22/33 for one full pass. Evaluate at0, every4096 episodes and final. Native
and HEFT starts use one shared selected policy, always report both. This is5
finite runs. If throughput makes online processing long, report measured ETA;
do not silently switch to16-step truncation or minibatch updates.

Final selected per-seed GiPH policies additionally get5 fixed sampling seeds;
average outcomes, never take per-case best across repetitions. All3 training
seeds and selection uncertainty are disclosed. A7epoch SFT comparison has unequal
training/inference costs; count episodes, moves, optimizer updates and time.
Primary500-base-macro paired ratios; supplementary862-config mean; by n/family
and physical feasibility, deterministic reference values and conditional paired
base bootstrap. Reports contain curves, params, mapping/order, hyperparameters,
source/data/code hashes, failures and negative outcomes.

## Execution and recovery

Finite local controller, two independent CPU method queues;
GiPH's three rate arms may execute concurrently, followed by its two confirmation
seeds concurrently. CAI's arms remain sequential. This permits at most4 actual
trainers (3GiPH +1CAI), each1Torch/OMP thread on the24GiB/10CPU host. No shared
policy/optimizer/RNG between arms. This changes wall time, not numerical work.
no unsupported chat cron. No external/cloud spending. Only trusted local input
and downloaded public pinned source, dependencies isolated in this run directory.
Checkpoint model/optimizer/RNG, input and code binding at committed update;
resume exact state, truncate uncommitted log/eval tails. Exclusive worker locks,
bounded2 retries and clear FAILURE files. Controller ends after both finite
tuning/confirmation queues and generates final report; no new SFT/RL experiments.

Engineering timing before launch: original DGL full2n forward/backward on12 real
TRAIN shapes averages4.11s/episode (0.36–17.73s range). This is a small balanced
shape probe, not a reliable full-run ETA. Parallelize independent arms rather than
reintroduce truncated episodes; publish ongoing measured episodes/time.

### TRAIN-input normalization correction before the retained formal run

Full17261 TRAIN audit found5 ratios (B-min)/(fast-min) >10, max128373,
including4 fast=min cases. This is not a VAL-quality based change. The initial
queue was stopped, including its already executed formal CAI updates, and all
artifacts/source snapshot archived under archive_unbounded_budget_ratio. No
completed formal GiPH update existed. The retained run normalizes added budget
input as (B-min)/(max-min), the physical feasible-cost range, and asserts[0,1].
No sample/label/algorithm changes; original GiPH inputs/behavior are unchanged.
Re-run fidelity/resume gates and restart all arms consistently. Do not count
the archived CAI trajectory as engineering or silently combine its observations
with the corrected run. Account its discarded formal computation separately.
