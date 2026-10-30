# What each file does

## Handoff documents

| File | Purpose |
|---|---|
| `START_HERE.md` | First document for the GPU collaborator; explains the package and reading order. |
| `PROJECT_IDEA.md` | Motivation, questions, study design, interpretation, related work, and future milestones. |
| `GPU_RUNBOOK.md` | Exact setup and execution commands, recovery guidance, and what to return. |
| `FILE_GUIDE.md` | This file map. |
| `README.md` | General installation and developer usage; includes data-download instructions for a fresh checkout. |
| `run_experiment.sh` | **The one command to run the study** on the GPU machine: installs what is missing, then runs every step in a tmux session. Subcommands `status`, `attach`, `stop`; running it again resumes. |
| `BUNDLE_INFO.json` | Only inside a ZIP built by `scripts/make_handoff.py`: file sizes and SHA-256 hashes. Not needed when the repository is shared through git. |

## Package and settings

| File | Purpose |
|---|---|
| `pyproject.toml` | Python package metadata, dependencies, optional quantization/test dependencies, CLI entry point, and pytest settings. |
| `configs/qwen_gender.yaml` | Full gender pilot: BF16/INT8/NF4, extraction sample, layer, strength grid, controls, and output root. |
| `configs/qwen_gender_gpu_check.yaml` | Small diagnostic on all three precisions and every vector source (216 validation questions) before spending GPU time on the full campaign. Its outputs are separate. |
| `configs/*.locked.json` | Created on the GPU machine by `steerquant lock`; captures exact checkpoint revision and resolved settings. |
| `.gitignore` | Excludes environments, capability-source downloads, credentials, caches, result folders, logs, and ZIPs from Git. BBQ data, prepared splits and the capability file are committed. |
| `.gitattributes` | Keeps data files byte-exact (their hashes are checked) and shell scripts LF-only, even when committing from Windows. |
| `LICENSE` | MIT license for the independently written pipeline. The dataset retains its separate license. |

## Main implementation: `src/steerquant/`

| File | Responsibility |
|---|---|
| `__init__.py` | Package description/version. |
| `__main__.py` | Allows `python -m steerquant` as an alternative to the installed command. |
| `cli.py` | Defines commands and dispatches requests to the modules below. Start here when following an execution path. |
| `config.py` | Loads defaults/YAML/JSON, validates settings, and fingerprints configurations. |
| `data.py` | Downloads official BBQ snapshots; normalizes examples; joins scoring targets; groups splits; selects extraction samples; constructs balanced, gold-respecting contrastive pairs; formats prompts; loads external capability QA. |
| `models.py` | Locks model revisions, loads the selected precision onto one device, applies chat templates, runs greedy generation, records runtime details, and checks that expected quantized modules were loaded. |
| `steering.py` | Captures block outputs, extracts/unit-normalizes vectors, calibrates activation scales, saves/checks vectors, creates random controls, and installs/removes intervention hooks. |
| `experiments.py` | Constructs validation conditions; runs/resumes evaluations; saves raw outputs; verifies run identities; chooses and freezes validation settings; applies those settings to held-out data. |
| `metrics.py` | Strict letter parsing; accuracy, invalidity, truncation, unknown and stereotypical-error rates; signed BBQ scores; group-pair metrics; answer-transition counts. |
| `analysis.py` | Paired scenario-cluster bootstrap intervals, vector geometry, CSV/JSON/Markdown reports, and validation response curves in SVG. |
| `io.py` | JSON/JSONL IO, atomic JSON writes, SHA-256 fingerprints, source/environment metadata, and overwrite protection. |
| `smoke.py` | Generates an entirely local tiny random model and synthetic questions, then exercises the complete workflow. Not a bias experiment. |

## Helpers and tests

| File | Purpose |
|---|---|
| `scripts/run_campaign.sh` | The pipeline that `run_experiment.sh` runs: environment check, tests, GPU check, gate, main validation, selection, test, capability, packaging. Skips finished stages, so it resumes. |
| `scripts/build_capability_qa.py` | Downloads MMLU and SQuAD 2.0 (pinned hashes) and rebuilds `data/capability/qa.jsonl` deterministically. Already run; only needed to reproduce or change the file. |
| `scripts/check_gate.py` | Go/no-go gate after the GPU check: all runs complete, finite vectors, and at most 5% invalid/truncated unsteered answers per precision. |
| `scripts/check_gpu.py` | Saves a GPU/package/environment diagnostic before downloading the full model, including PyTorch architecture support and tiny real BF16/INT8/NF4 kernel runs. |
| `scripts/package_results.py` | Packages artifacts, reports, logs, locked configs, and data provenance to return to the study owner. Excludes environments and model caches. |
| `scripts/make_handoff.py` | Rebuilds this source/data/instructions ZIP and its per-file hash inventory. |
| `tests/conftest.py` | Offline test environment and synthetic dataset fixtures. |
| `tests/test_data.py` | Group isolation, target correctness, answer-position balance, metadata validation, and manifest tampering. |
| `tests/test_metrics.py` | Strict parsing and hand-computed bias/accuracy/abstention/transition checks. |
| `tests/test_steering.py` | Block-output indexing, zero-strength behavior, last-position effects, cleanup, and cached-generation coverage. |
| `tests/test_pipeline.py` | End-to-end offline workflow, frozen selection, resume/integrity checks, control conditions, and bootstrap units. |
| `tests/test_handoff.py` | Archive completeness/checksums, exclusion of environments/model weights, partial-result packaging, and credential-free diagnostics. |
| `.github/workflows/tests.yml` | Runs the offline suite if the repo is later hosted on GitHub. It has not been executed on GitHub as part of this handoff. |

## Data in this ZIP

| File/folder | Purpose |
|---|---|
| `data/raw/bbq/Gender_identity.jsonl` | Raw gender-related BBQ examples used for the first task. |
| `data/raw/bbq/Age.jsonl`, `Race_ethnicity.jsonl` | Official raw data for later extensions; not part of the first run. |
| `data/raw/bbq/additional_metadata.csv` | Official bias-scoring targets joined by category/scenario/example. |
| `data/raw/bbq/source.json` | Upstream repository commit, download URLs, and raw file hashes. |
| `data/raw/bbq/LICENSE` | BBQ dataset's CC BY 4.0 license. |
| `data/prepared/gender/train.jsonl` | Group-separated pool used to select the vector-extraction examples. |
| `data/prepared/gender/validation.jsonl` | Layer/strength/control evaluation and selection. |
| `data/prepared/gender/test.jsonl` | Held-out final evaluation at frozen settings. |
| `data/prepared/gender/manifest.json` | Split hashes, counts, grouping and stratification definition, identity mix per split, and annotation coverage. |
| `data/capability/qa.jsonl` | 2,000 held-out, demographically neutral questions: 1,000 MMLU, 500 answerable and 500 unanswerable SQuAD 2.0. Evaluated at frozen settings after the test split. |
| `data/capability/manifest.json`, `README.md`, `LICENSE-MMLU.txt` | Sources, hashes, construction rules, per-part counts, and licenses of the capability file. |
| `examples/neutral_qa.jsonl` | Three formatting examples for external capability/uncertainty data; not an adequate evaluation suite. |

## Detailed documentation

| File | Purpose |
|---|---|
| `docs/protocol.md` | Exact scientific and implementation conventions, assumptions, limits, and future work. |
| `docs/verification.md` | What has been tested and what still requires GPU execution. |
| `docs/data-audit.json` | Actual upstream/split hashes and the training-pair audit. |
| `docs/local-environment.txt` | Mac CPU environment versions used for verification; not a CUDA installation lockfile. |

## Files created by a run

| Location | Contents |
|---|---|
| `artifacts/gpu-check/` | Small all-precision diagnostic vectors, validation predictions, and an exercise selection file. |
| `artifacts/qwen-gender-seed42/vectors/` | `bf16.pt`, `int8.pt`, `nf4.pt`, and matching JSON metadata. These are steering vectors, not full model checkpoints. |
| `artifacts/qwen-gender-seed42/runs/validation/` | One folder per precision/source/layer/strength condition. |
| `artifacts/qwen-gender-seed42/selection.json` | Frozen validation-selected settings and unchanged-reference transfer conditions. |
| `artifacts/qwen-gender-seed42/runs/test/` | Held-out predictions for the frozen conditions. |
| `artifacts/qwen-gender-seed42/runs/capability/` | Held-out capability/neutral-uncertainty QA results. |
| `runs/.../run.json` | A condition's identity, expected size, completion status, runtime/package/source metadata, and output hashes. |
| `runs/.../predictions.jsonl` | Original example metadata, generated text/tokens, parsed prediction, and truncation flag. |
| `runs/.../metrics.json` | That condition's computed metrics. |
| `reports/` | `metrics.csv`, `analysis.json`, `report.md`, validation `response.svg`, and for capability `capability_by_category.csv`. |
| `logs/` | Setup, GPU diagnostic, and per-stage run logs. The runner adds `campaign.log` (everything), `campaign-status.txt` (current stage or DONE/FAILED/STOPPED), `gate.log`/`gate.json`, `.done/` stage markers, and lock files. |
| `steering-results-<time>.zip` | Results archive created at the end of an unattended run. |

Full model weights are downloaded into the Hugging Face cache. Keep that cache outside `artifacts/` and outside the returned ZIP.
