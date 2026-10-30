# GPU runbook

These commands assume Bash on an Ubuntu machine with an NVIDIA GPU of at least 80 GB (a 96 GB Blackwell card is fine). Run them from the `steering-quantization` folder. Model downloads need internet access; bundled BBQ data does not.

The GPU collaborator's job is to execute the fixed pipeline and return all artifacts. No final bias-reduction result is assumed in advance.

**Short version:** do sections 0 and 1 once, then run `bash scripts/start_campaign.sh` (section 2). Everything else runs unattended in the background, and you can turn off the computer you connected from.

## 0. Prepare the machine (once)

- **Stop the machine from suspending.** A suspended machine pauses the run. On Ubuntu Desktop, open Settings → Power and set *Automatic Suspend* to Off. On a server, or to be certain, run `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`; `unmask` undoes it. `start_campaign.sh` also blocks suspend while it runs, when the system allows it.
- **Disk:** keep at least 50 GB free. The model (about 15 GB) goes into the Hugging Face cache, `~/.cache/huggingface` by default, which is outside the project. To use another disk, add `export HF_HOME=/path/to/big/disk/hf-cache` to `~/.bashrc` and open a new terminal before starting.
- **No Hugging Face token is needed.** `Qwen/Qwen2.5-7B-Instruct` is public and not gated. A token (`export HF_TOKEN=...`) only raises download rate limits. It is never written into artifacts.
- Do not run other jobs on the GPU during the campaign.

## 1. Create the environment (once)

```bash
cd /path/to/steering-quantization
sudo apt install -y python3-venv          # only if venv creation fails
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
nvidia-smi                                 # Blackwell needs an NVIDIA driver from the 570 series or newer
```

Install a CUDA build of PyTorch **for CUDA 12.8 or newer**; older builds have no Blackwell kernels. For example:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If the [PyTorch selector](https://pytorch.org/get-started/locally/) offers a newer CUDA version that your driver supports, that also works. Then install the project and check the machine:

```bash
python -m pip install -e '.[dev,quant]'
python -m pip check
python scripts/check_gpu.py --output logs/gpu_environment.json
```

Expected result: `"ready_for_gpu_check": true`, your GPU and its compute capability (12.0 for RTX PRO 6000 Blackwell), and `kernel_checks` with `bf16_matmul`, `int8_linear` and `nf4_linear` all `true`. These run tiny real BF16, LLM.int8() and NF4 layers on the GPU. If it reports *no kernels for sm_XXX*, the PyTorch build is too old for the GPU: reinstall it from a newer CUDA index. If an INT8/NF4 kernel check fails, upgrade bitsandbytes (`python -m pip install -U bitsandbytes`) and rerun the check.

The recorded Mac environment in `docs/local-environment.txt` is for CPU verification only. It is not a CUDA lockfile. If the machine cannot support a requested precision, return the diagnostic instead of silently changing experiment precision.

## 2. Run everything unattended (recommended)

```bash
source .venv/bin/activate
bash scripts/start_campaign.sh
```

This starts `scripts/run_campaign.sh` in the background, in its own session, so it is not tied to your terminal or SSH connection. Once it prints *Campaign running in the background*, you can close the terminal, disconnect, and turn off your laptop. The job runs on the GPU machine. In order, it:

1. Checks the GPU environment, runs the offline test suite (32 tests) and the CPU smoke test.
2. Runs the **GPU check** on all three precisions (BF16, LLM.int8(), NF4) with every vector source and control, on 216 validation questions: 14 conditions, 3,024 generations.
3. Applies the **go/no-go gate** (section 3). It stops here if the gate fails.
4. Runs the **main campaign**: locks the model revision, extracts vectors and runs the full validation sweep for BF16, then INT8, then NF4. That is 54 conditions × 1,240 questions = 66,960 generations.
5. Writes the validation report, **freezes the selection**, runs the held-out **test** split once at the frozen settings, and writes the test report.
6. Optionally runs the capability suite (section 5).
7. Packages everything into `steering-results-<UTC time>.zip` in the project folder.

No runtime has been measured yet. Expect many hours; LLM.int8() is the slowest precision.

Check on it later, from any SSH session:

```bash
cat logs/campaign-status.txt     # RUNNING <stage> / STOPPED ... / FAILED in stage ... / DONE
tail -n 30 logs/campaign.log     # recent progress; evaluation prints a line every 25 questions
tail -f logs/campaign.log        # live view; Ctrl+C stops watching, not the job
```

Each stage also has its own log at `logs/<stage>.log`.

- **To stop it:** `bash scripts/stop_campaign.sh`.
- **To resume** after stopping, a failure, a reboot or a power cut, run `bash scripts/start_campaign.sh` again. Finished stages are skipped. Interrupted evaluations continue where they stopped (`--resume`). A finished vector, lock file, selection or report is never recomputed or overwritten.
- If the status still says `RUNNING` after a reboot, the job was killed by the reboot. Start it again.

Do not edit source, configs, package versions or prepared data between starting and finishing. Those changes alter run identities and comparability. If a change is unavoidable, use a fresh artifact root and document it as a new campaign.

## 3. The GPU-check gate

`scripts/check_gate.py` decides whether the long campaign may start. It passes only if, **for every precision**:

- all GPU-check runs completed;
- extracted vector norms and activation scales are finite and positive;
- the **unsteered** model's answers are unparseable in at most 5% of questions, and hit the 12-token cap in at most 5%.

Why: only a standalone letter (`B`, `(B)`, `B.`) counts as an answer. Anything else is invalid and scored as wrong. If the plain model often answers like `B. The nurse` or `The answer is B`, every accuracy number would mostly measure answer formatting. Steered conditions are reported but never gated: their invalid rates are results. The gate reads only validation questions, never the test split.

If the campaign stops at the gate (`STOPPED at the GPU-check gate`):

1. Read `logs/gate.log`. It shows each condition's invalid and truncation rates and its most common raw outputs. `logs/gate.json` and the GPU-check `predictions.jsonl` files have every output.
2. **If the format problem is real,** fix the prompt instruction or the answer parser *before* the main campaign. That is a protocol change, so write it down. Then delete the GPU-check outputs and run it again:
   ```bash
   rm -rf artifacts/gpu-check reports/gpu-check logs/.done/gpu-check-*
   bash scripts/start_campaign.sh
   ```
3. **If you judge the rate acceptable,** continue and record that decision: `SKIP_GATE=1 bash scripts/start_campaign.sh`. Thresholds can also be changed explicitly, e.g. `MAX_INVALID=0.10 bash scripts/start_campaign.sh`.

Do not use GPU-check accuracy or debiasing scores to redesign the main campaign. The gate is about whether answers can be scored, not about how good they are.

## 4. Manual step-by-step (alternative to section 2)

Use this only if you want to drive each stage yourself. Run it inside `tmux` so it survives disconnects: `tmux new -s steer` to start, Ctrl+B then D to detach, `tmux attach -t steer` to come back. The machine must not suspend (section 0).

The ZIP already contains `data/raw/bbq/` and `data/prepared/gender/`. Do not re-run `fetch-bbq` or `prepare` into those folders, and do not shuffle or recreate the splits.

```bash
mkdir -p logs; set -o pipefail
pytest -q 2>&1 | tee logs/tests.log
steerquant smoke --output artifacts/offline-smoke 2>&1 | tee logs/offline-smoke.log
steerquant plan --config configs/qwen_gender.yaml 2>&1 | tee logs/main-plan.log
```

`plan` validates split hashes. Expected full validation budget: 14 BF16, 20 INT8 and 20 NF4 conditions, **66,960 generations** over 1,240 validation questions. The extraction sample is 128 examples from the 3,192-example training pool. Validation and test each have 1,240 questions. These counts include many variants of a smaller number of scenarios.

GPU check on all precisions, then the gate:

```bash
steerquant lock --config configs/qwen_gender_gpu_check.yaml --output configs/qwen_gender_gpu_check.locked.json
for p in bf16 int8 nf4; do
  steerquant extract --config configs/qwen_gender_gpu_check.locked.json --precision $p 2>&1 | tee logs/gpu-check-$p-extract.log
done
for p in bf16 int8 nf4; do
  steerquant evaluate --config configs/qwen_gender_gpu_check.locked.json --precision $p --split validation 2>&1 | tee logs/gpu-check-$p-validation.log
done
steerquant summarize --config configs/qwen_gender_gpu_check.locked.json --split validation --output reports/gpu-check
python scripts/check_gate.py --config configs/qwen_gender_gpu_check.locked.json 2>&1 | tee logs/gate.log
```

Main campaign. The full configuration has its own artifact root; GPU-check vectors cannot be substituted.

```bash
steerquant lock --config configs/qwen_gender.yaml --output configs/qwen_gender.locked.json
for p in bf16 int8 nf4; do
  steerquant extract --config configs/qwen_gender.locked.json --precision $p 2>&1 | tee logs/main-$p-extract.log
  steerquant evaluate --config configs/qwen_gender.locked.json --precision $p --split validation 2>&1 | tee logs/main-$p-validation.log
done
```

Only after all validation conditions finish, freeze settings and evaluate the held-out test split:

```bash
steerquant summarize --config configs/qwen_gender.locked.json --split validation --output reports/gender-validation
steerquant select --config configs/qwen_gender.locked.json --output artifacts/qwen-gender-seed42/selection.json \
  --accuracy-budget 0.02 --invalid-budget 0.01 --unknown-budget 0.02 2>&1 | tee logs/main-selection.log
steerquant evaluate --config configs/qwen_gender.locked.json --precision all --split test \
  --selection artifacts/qwen-gender-seed42/selection.json 2>&1 | tee logs/main-test.log
steerquant summarize --config configs/qwen_gender.locked.json --split test --output reports/gender-test
```

Models are loaded and released by each command. Do not launch the same condition twice concurrently into the same artifact root. Selection records any source with no admissible setting. That is a result, not an error to tune away. Reference settings are also applied unchanged to quantized receivers. Never select strengths by test scores.

## 5. Optional capability/neutral-uncertainty evaluation

This handoff does **not** contain an adequate capability benchmark; `examples/neutral_qa.jsonl` only shows the format. Choose a properly sized held-out suite before claiming capability preservation. The QA file needs unique IDs, scenario groups, a question, 2–4 answer choices and a zero-based gold answer. Record its source, license and construction, and keep it unchanged between precisions.

With the unattended runner, pass it at start: `QA_FILE=/path/to/held_out_qa.jsonl bash scripts/start_campaign.sh`. If the campaign has already finished, the same command runs only the capability stage and packages a new results ZIP. Manually:

```bash
steerquant evaluate --config configs/qwen_gender.locked.json --precision all --split capability \
  --qa-file /path/to/held_out_qa.jsonl --selection artifacts/qwen-gender-seed42/selection.json 2>&1 | tee logs/main-capability.log
steerquant summarize --config configs/qwen_gender.locked.json --split capability --output reports/gender-capability
```

## 6. Return the results

The unattended runner creates `steering-results-<UTC time>.zip` at the end. Manually: `python scripts/package_results.py --output steering-results.zip`. It contains both artifact roots, reports, logs, locked configs, data provenance and a hash inventory. It excludes the environment, the model cache and synthetic smoke outputs. Pass `--artifact-root path` for each nondefault artifact root.

Include a short note: whether all stages finished, the GPU type, and any failures or intentional deviations (including a `SKIP_GATE` decision). A failed or partial run is worth returning. Raw predictions and vector metadata are needed to audit the study, so do not send only CSV rows or screenshots.

## Interrupted runs and common problems

| Situation | Action |
|---|---|
| Status says `FAILED in stage X` | Read `logs/X.log`, fix the cause, run `bash scripts/start_campaign.sh` again. |
| Status says `STOPPED at the GPU-check gate` | See section 3. |
| Status says `RUNNING` but nothing runs (after reboot) | Run `bash scripts/start_campaign.sh` again; it resumes. |
| "A campaign is already running here" | It is still running; watch `logs/campaign.log`. Stop it with `scripts/stop_campaign.sh` if needed. |
| No CUDA GPU visible | Check driver/PyTorch CUDA build. Save `logs/gpu_environment.json`. |
| "No kernels for sm_XXX" | PyTorch build too old for the GPU: reinstall from a CUDA 12.8+ index (section 1). |
| No BF16 support | Return the diagnostic; do not relabel FP16 as BF16. |
| bitsandbytes or CUDA kernel errors | Upgrade bitsandbytes, rerun `check_gpu.py`; save the exact error and `pip freeze`. |
| Model download blocked | Arrange access to the public checkpoint; keep the locked revision. |
| Out of GPU memory | Check `nvidia-smi` for other processes. Keep the pipeline settings; return the error if unresolved. |
| Resume fails on a malformed prediction line after a power cut | The last line of that run's `predictions.jsonl` was cut mid-write. Delete only that incomplete last line, then resume. |
| Vector file exists | Never re-extract a finished vector. A leftover `.tmp` file is not a finished vector; resuming retries it. |
| Config/package/source changed | Use a fresh artifact root and a clearly documented new campaign. |
| Invalid outputs or poor debiasing in the main run | Preserve them. They are outcomes, not permission to change parsing or remove examples. |

The held-out study is still an exploratory single-model/single-seed pilot. Activation quantization, repeated seeds, broader capability controls, and additional categories/models are subsequent work.
