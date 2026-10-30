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
- GPU diagnostic (including real INT8/NF4 kernel checks), go/no-go gate, results-packaging scripts, and an unattended runner that survives disconnects and resumes after interruptions.
- A SHA-256 inventory of the bundle, in `BUNDLE_INFO.json` in the ZIP.

Model weights, Python environments, credentials, prior synthetic outputs, and Git history are not included. The initial model download needs internet access. The bundled raw data retains its own CC BY 4.0 license; the independently written pipeline has the MIT license in the repository root.

## What has already been checked

- 32 tests pass in the documented local CPU environment.
- A random tiny-transformer smoke test completes extraction, validation, selection, held-out testing, capability evaluation, and reporting without downloading a model.
- Real gender data splits have 3,192 training-pool examples (27 scenarios), 1,240 validation examples (11 scenarios), and 1,240 test examples (12 scenarios), with no shared scenario groups. The splits are stratified by identity type, so each split is about 81–82% name-based, 14–16% trans/non-trans, and 3% gender-term scenarios.
- The pilot extraction sample has 128 examples across 27 scenarios. All positive targets match gold answers, and answer letters are balanced.

The Qwen GPU runs and bitsandbytes CUDA paths still need to be exercised. CPU smoke scores are not study results. The full test suite has not been re-run since the re-split and the new scripts; the runner runs it first on the GPU machine and stops if anything fails.

## First actions

Copy or unzip the folder to the GPU machine and open a terminal in it. Follow sections 0–1 of [GPU_RUNBOOK.md](GPU_RUNBOOK.md) once (machine settings, environment, GPU check), then start everything with:

```bash
bash scripts/start_campaign.sh
```

It runs the tests, a small GPU check on all three precisions, a go/no-go gate, the main validation sweep, frozen selection, the held-out test, and packaging. It keeps running after you disconnect, and resumes if started again after an interruption. If the run fails, return the logs and partial artifacts instead of changing the scientific settings to make the errors disappear.

The original `README.md` is the general developer guide. For this bundled handoff, use the runbook as the main execution guide: data preparation is already complete, so downloading/preparing the same data again is unnecessary.
