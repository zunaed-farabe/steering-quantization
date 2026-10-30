# Steering under weight quantization

A research pipeline for studying whether bias-reduction activation steering transfers from a floating-point language model to its weight-quantized versions. The first task is evidence-based answering on gender-related BBQ questions. There are **no real-model experimental results yet**.

For the collaborator ZIP, begin with [START_HERE.md](START_HERE.md), then [the project idea](PROJECT_IDEA.md), [GPU runbook](GPU_RUNBOOK.md), and [file guide](FILE_GUIDE.md). The ZIP includes audited prepared data and a smaller GPU-check configuration.

## Experimental scope

- Reference: Qwen2.5-7B-Instruct in BF16; additional receivers: bitsandbytes LLM.int8() and 4-bit NF4.
- Contrastive mean-difference vectors, unit-normalized, extracted separately at each precision.
- Original-vector transfer, native-vector extraction, deterministic norm-matched random directions, prompt-only control, and unsteered baselines.
- Original reference layer/strength copied unchanged to quantized receivers, plus independently validation-selected strengths.
- Gold-respecting contrastive pairs, grouped data splits, raw predictions, explicit validity/abstention metrics, and paired scenario-cluster confidence intervals.

LLM.int8() includes mixed-precision outlier handling and internally quantized matmul inputs; it is **not** a pure uniform W8A16 arithmetic experiment. NF4 and LLM.int8() differ in algorithm as well as bit width. Report these as deployment schemes, not an isolated causal effect of bit width. Residual-stream activations and steering interventions remain floating point. Independently controlled activation quantization and PCA extraction are later milestones.

## Install

Use Python 3.10+ in a virtual environment. A Linux CUDA machine is the supported target for the quantized experiments; each process uses one explicit device. An 80 GB GPU can be selected when the execution environment is decided. Package installation itself does not download model weights.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
# On the chosen CUDA machine, also install quantization support:
python -m pip install -e '.[quant]'
```

Install a PyTorch build appropriate for that machine's CUDA runtime before launching GPU jobs. The exact locally verified versions are recorded in `docs/local-environment.txt`; they are a Mac CPU test environment, not a validated CUDA lockfile. Every experiment records installed package versions and runtime quantization settings.

## Verify locally without network or a GPU

```bash
pytest -q
steerquant smoke --output artifacts/offline-smoke
```

The smoke command creates a tiny, randomly initialized Llama architecture and synthetic questions. It exercises extraction, generation hooks, validation selection, frozen test evaluation, and report generation. **Its scores say nothing about bias, fairness, or model capabilities.** Tests create all checkpoints locally and run with Hugging Face offline mode.

## Prepare the actual experiment

Run commands from this repository root. Configuration paths are relative to the launch working directory.

```bash
steerquant fetch-bbq --output data/raw/bbq --axes gender
steerquant prepare \
  --raw data/raw/bbq/Gender_identity.jsonl \
  --metadata data/raw/bbq/additional_metadata.csv \
  --output data/prepared/gender --seed 42
steerquant plan --config configs/qwen_gender.yaml
steerquant lock --config configs/qwen_gender.yaml \
  --output configs/qwen_gender.locked.json
```

`fetch-bbq` resolves and records the official BBQ commit, file URLs, hashes, and dataset license. `prepare` uses official scoring metadata rather than inferring bias targets from correctness. It keeps every `(category, question_index)` group in one split. For gender, groups are first stratified by identity type: proper names (M/F), gender terms (man/woman, boy/girl), and trans/non-trans. Within each type, groups are assigned largest first to whichever split is furthest below its 60/20/20 row target, so every split has a similar identity mix. Other categories are not stratified until a matching definition is added in `data.identity_stratum`. The bundled gender split has 3,192/1,240/1,240 examples in 27/11/12 scenarios. The manifest records counts, identity mix, and annotation coverage. Sixteen gender training items in the checked upstream snapshot lack matching target annotations; they remain eligible for gold-based extraction, and missing-target coverage is explicit in scoring.

Do not generate new splits after inspecting test results. Small `train_limit` values sample across training groups and evidence/polarity strata. Optional validation/test limits retain complete scenario groups; their example counts are approximate budgets.

`lock` resolves a full Hugging Face checkpoint commit before model runs. Use `HF_TOKEN` or the Hugging Face login mechanism if a later checkpoint requires access; credentials are never written into artifacts. Model downloads occur when extraction/evaluation first loads the locked model.

## Run on the GPU machine

For an unattended run that survives disconnects and resumes after interruptions, use `bash scripts/start_campaign.sh`. [GPU_RUNBOOK.md](GPU_RUNBOOK.md) describes it, together with the go/no-go gate it applies after a small all-precision GPU check. The individual commands are:

```bash
steerquant extract --config configs/qwen_gender.locked.json --precision all
steerquant evaluate --config configs/qwen_gender.locked.json \
  --precision all --split validation
steerquant select --config configs/qwen_gender.locked.json \
  --output artifacts/qwen-gender-seed42/selection.json \
  --accuracy-budget 0.02 --invalid-budget 0.01 --unknown-budget 0.02
steerquant evaluate --config configs/qwen_gender.locked.json \
  --precision all --split test \
  --selection artifacts/qwen-gender-seed42/selection.json
steerquant summarize --config configs/qwen_gender.locked.json \
  --split validation --output reports/gender-validation
steerquant summarize --config configs/qwen_gender.locked.json \
  --split test --output reports/gender-test
```

You may run one precision at a time on the same GPU or assign separate processes/devices to different precisions. Reference vectors must exist before evaluation. Use a separate artifact root for every config/seed; do not edit an experiment config after extraction. All validation cells must finish before selection. `evaluate --resume` verifies completed runs and skips them, or continues an interrupted run. A malformed partial JSONL record fails closed and requires inspection rather than silently dropping a question. Dataset, config, vector, or source changes can create new run identities; keep a new artifact root for changed experiments.

Selection maximizes ambiguous-context accuracy subject to disambiguated-accuracy, unnecessary-unknown, invalid-response, and truncation budgets relative to **each receiver's own baseline**. These are exploratory pilot defaults; decide and freeze confirmatory budgets after the pilot. A source with no admissible setting is explicitly recorded. Unchanged reference settings are still tested on quantized receivers even when they would fail those receiver-side budgets.

## Capability and neutral uncertainty controls

Supply a genuinely held-out JSONL file with `id`, `group`, `question`, `answers` (2–4 strings), and a zero-based `gold`; `context` and `category` are optional. The group field controls the bootstrap unit. Include both answerable and unanswerable demographically neutral contexts. A formatting example is included in `examples/neutral_qa.jsonl`; its three questions are **not an adequate benchmark**.

```bash
steerquant evaluate --config configs/qwen_gender.locked.json \
  --split capability --qa-file /path/to/held_out_qa.jsonl \
  --selection artifacts/qwen-gender-seed42/selection.json
steerquant summarize --config configs/qwen_gender.locked.json \
  --split capability --output reports/gender-capability
```

Capability evaluation uses the same frozen selected settings and keeps steering active while answering. It never chooses settings using capability test answers. Use distinct experiment roots if comparing different capability suites, or combine them into one file with category labels.

## Artifacts

Each experiment stores vectors plus metadata, the frozen selection, and one prediction JSONL/manifest/metrics JSON per condition. Artifacts include checkpoint revision, data hashes, chat-template hash, extraction examples, normalization, reference activation scale, quantization configuration, module/parameter types, package versions, source fingerprint, and GPU information when applicable.

Reports contain `metrics.csv`, `analysis.json`, `report.md`, and a validation `response.svg`. JSON includes ambiguity-separated metrics, annotation/validity denominators, group-pair outcomes, answer transitions, and paired intervals versus each precision's baseline. Rates are fractions, not percentages. Confidence intervals are descriptive and conditional on one extraction seed. They are not corrected for multiple comparisons or sufficient evidence of equivalence. Repeat extraction across seeds and models before making general claims; hierarchical pooling and equivalence testing are not implemented in v1.

See [the protocol](docs/protocol.md) for exact semantics and [the verification record](docs/verification.md) for what has and has not been run.

## Attribution

This implementation is written independently; it does not import or copy code from the reference checkout. Cite the source methods and datasets in any resulting paper:

- [BBQ](https://github.com/nyu-mll/BBQ), [paper](https://arxiv.org/abs/2110.08193), CC BY 4.0 data.
- [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681).
- [Shifting Perspectives](https://arxiv.org/abs/2503.05371), a methodological/reproduction reference.
- [Steering Under Compression](https://arxiv.org/abs/2609.06473), related quantization study.
- [Investigating Social Bias Changes in Quantized Language Models](https://arxiv.org/abs/2602.06181).

BBQ is an English benchmark grounded primarily in U.S. social contexts. Benchmark improvements do not establish universal fairness or removal of bias from a model.
