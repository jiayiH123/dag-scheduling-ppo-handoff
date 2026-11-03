# WeCAN reproduction status: author implementation not verified available

Paper: Ruisong Zhou et al., A Learning Method with Gap-Aware Generation for
Heterogeneous DAG Scheduling, arXiv2603.23249v2.
https://arxiv.org/html/2603.23249v2
https://haijunzou-sms.github.io/publications/

Checked2026-11-03: paper full text, author's publication page, GitHub repository
searches for the title/arXivID and WeCAN+scheduling. No verifiable official
implementation was found. This is an explicit missing implementation, not a
claim that all possible private/unindexed implementations are absent.

Included: ../../papers/WeCAN_v2.pdf and WeCAN_v2_latex.tar, provenance/hashes in
papers/SOURCES.json; research notes in docs/wecan2026.md. The LaTeX archive is
paper source, NOT runnable model code. No WeCAN benchmark result is fabricated.

To reproduce: implement weighted compatibility-aware cross-attention, LDDGNN,
single-pass task/pool scores and global skip parameters, the decreasing skip
generation map and its training objective following sectionsIV and AppendixF.
Record graph/platform datasets, the paper's multiple sampled candidates and
five inference seeds separately from one-forward deterministic deployment.

Necessary shared-problem adaptations remain to design/test: directed inter-pool
communication, additive monetary hard budget and suffix reserve, cumulative
pool capacity, and feasibility preservation when skipping/waiting. The paper
lists communication as future work; its unchanged scores are not yet a valid
baseline for our problem. Use existing daglab/state_score.py and
daglab/external_baseline_sim.py as physical interfaces, not as claimed WeCAN.
