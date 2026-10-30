# Start here — GPU handoff

This package contains the experimental code, prepared data, and instructions for a study of **bias-reduction activation steering under model quantization**.

**The immediate request:** verify the pipeline on a Linux NVIDIA GPU, then run the first gender-bias experiment. A GPU with at least 80 GB is the intended environment. No real-model results have been collected yet.

## Read in this order

1. [Project idea](PROJECT_IDEA.md): what we want to learn, the experiment, and how to interpret it.
2. [GPU runbook](GPU_RUNBOOK.md): copyable commands, from installation to returning results.
3. [File guide](FILE_GUIDE.md): what every code/configuration/documentation file does.
4. [Detailed protocol](docs/protocol.md): exact extraction, intervention, scoring, and selection rules.

## What is included

- Independently implemented Python research package, CLI, and 32 offline tests (29 pipeline tests plus 3 handoff/diagnostic checks).
- Main experiment config and a smaller GPU-check config covering BF16, INT8 and NF4.
- Official BBQ gender, age, and race raw data; scoring metadata; provenance and dataset license.
- Ready-to-use, audited gender train/validation/test files and manifest. The initial experiment uses only gender.
- A held-out capability/neutral-uncertainty suite: 2,000 demographically neutral questions from MMLU and SQuAD 2.0 (`data/capability/`).
- GPU diagnostic (including real INT8/NF4 kernel checks), go/no-go gate, results packaging, and `run_experiment.sh`: one command that sets up the machine and runs the whole study in tmux, resuming after interruptions.

Model weights, Python environments and credentials are not included. The initial model download needs internet access. The BBQ data keeps its CC BY 4.0 license, MMLU its MIT license, and the SQuAD-derived rows CC BY-SA 4.0. The independently written pipeline has the MIT license in the repository root.

## What has already been checked

- 32 tests pass in the documented local CPU environment.
- A random tiny-transformer smoke test completes extraction, validation, selection, held-out testing, capability evaluation, and reporting without downloading a model.
- Real gender data splits have 3,192 training-pool examples (27 scenarios), 1,240 validation examples (11 scenarios), and 1,240 test examples (12 scenarios), with no shared scenario groups. The splits are stratified by identity type, so each split is about 81–82% name-based, 14–16% trans/non-trans, and 3% gender-term scenarios.
- The pilot extraction sample has 128 examples across 27 scenarios. All positive targets match gold answers, and answer letters are balanced.
- The capability file passes the project's loader, has no duplicate questions, has no demographic terms in questions or options, and rebuilds byte-identically from its pinned sources.

The Qwen GPU runs and bitsandbytes CUDA paths still need to be exercised. CPU smoke scores are not study results. The full test suite has not been re-run since the re-split, the capability additions and the new scripts. The runner runs it first on the GPU machine and stops if anything fails.

## First actions

On the Ubuntu GPU machine, do [GPU_RUNBOOK.md](GPU_RUNBOOK.md) section 0 once (turn off automatic suspend, check disk space and `nvidia-smi`). Then:

```bash
git clone <your private repo URL> steering-quantization   # or `git pull` in an existing checkout
cd steering-quantization
bash run_experiment.sh
```

It installs what is missing, then in a tmux session runs, in order:

1. the tests;
2. a small GPU check on all three precisions;
3. the go/no-go gate;
4. the main validation sweep;
5. the frozen selection;
6. the held-out test;
7. the capability suite;
8. packaging.

Leave the view with Ctrl+B, then D; it keeps running after you disconnect. Run `bash run_experiment.sh status` to check on it, and `bash run_experiment.sh` again to resume after any interruption. If the run fails, return the logs and partial artifacts instead of changing the scientific settings to make the errors disappear.

The original `README.md` is the general developer guide. For running the study, use the runbook. Data preparation is already complete, so downloading or preparing the same data again is unnecessary.
