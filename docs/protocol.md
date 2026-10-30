# v1 experimental protocol

Status: exploratory pilot. Main task: evidence-based demographic QA. Weight-only quantization first; activation quantization next.

## Research questions

1. Do original-model steering vectors retain their functional effect after weight quantization?
2. Can native extraction or strength recalibration recover a change in steering effectiveness?
3. Does vector geometry track functional transfer, abstention, or capability loss?

Avoid presupposing that quantization must degrade steering. Quantization can change the baseline itself. Compare interventions to the receiver's baseline and also report absolute performance.

## Dataset and target

Normalize official BBQ examples and join official `target_loc` scoring metadata by category, question_index, and example_id. Bias target means the stereotype-consistent *answer to the question*, including question polarity; it is not simply the named stereotyped group or the correct answer. Unmatched annotations have null targets and explicit coverage. They are never inferred from the gold label.

Group splitting operates on `(category, question_index)`. This blocks the known family of scenario variants from crossing splits; it does not prove the absence of all semantic overlap or benchmark contamination in model pretraining. Gender groups are stratified by identity type: proper-name templates (M/F), explicit gender terms, and trans/non-trans templates, with a group's type being its majority label kind. Within each type, groups are shuffled with seed 42, ordered largest first, and each is assigned to the split furthest below its 60/20/20 row target. The result is 3,192/1,240/1,240 examples in 27/11/12 scenarios, each split about 81–82% names, 14–16% trans/non-trans, and 3% terms. Within the trans type, validation holds one name-based trans scenario and test holds two term-based ones; seven trans scenarios cannot be divided more evenly. This split replaced an earlier unstratified one before any model was run. Test labels are used only in final evaluation after validation selection.

For extraction, ambiguous examples pair the gold unknown answer with each incorrect named attribution. Disambiguated examples pair the evidence-supported gold answer with the other named attribution. A correct answer naming the stereotyped group remains positive. Three cyclic answer permutations balance each semantic answer over A/B/C. Average differences within each example, then within ambiguity condition, then average the two conditions equally. Training-row subsampling is distributed across groups and evidence/polarity strata.

This construction deliberately learns **evidence following**. It may capture generic correctness or uncertainty rather than a bias-specific feature. Random-vector, prompt-only, generic QA, neutral uncertainty, and subgroup checks constrain interpretation; they do not establish a unique causal representation of fairness. An independently extracted generic-uncertainty direction is a planned additional control, not included in v1.

## Extraction and injection

- Supported architectures: Transformers Qwen2, Llama, Mistral with explicit `model.layers` adapters.
- Layer index is zero-based decoder block OUTPUT, before the next block/final normalization.
- Extraction uses the last answer token after appending positive/negative answer letters to the same rendered chat prompt. The positive-minus-negative mean is accumulated in FP32 and unit-normalized.
- Calibration is the average L2 norm of the unsteered final **prompt** token at that block over extraction examples.
- Intervention is `h_last += alpha * reference_activation_scale * unit_vector`.
- The scale comes from the floating-point reference for all precisions and vector sources, so extraction-norm changes do not silently change the intervention dose.
- On prefill, modify the final prompt position so the first answer is affected. During cached generation, modify the current final position at every step. Earlier prompt positions are not modified.
- Zero strength installs no hook. All hooks are removed in a finally block, including failures.
- Random directions are reproducible from extraction seed/layer and use identical unit norm/dose.

Neither earlier-block prompt modifications nor multi-layer superposition are implied by v1. A list of layers means separate single-layer conditions.

## Precisions and implementation boundaries

BF16 baseline uses floating-point model weights. NF4 uses bitsandbytes 4-bit normal-float weights, no double quantization, and configured floating-point compute. LLM.int8() uses its mixed-precision outlier decomposition with threshold 6.0 and internally quantized matmul inputs. It is not a strict uniform W8A16 control. Residual activations/interventions remain floating point; v1 does not independently vary activation precision. No CPU offload or model sharding is used in v1. Models are not cast to half/BF16 after quantization. Runtime module counts, parameter dtypes, and quantization settings are saved; a requested quantized run fails if the expected quantized linear modules are absent.

Quantization scheme and bit width are confounded in this first deployment comparison. A later matched-quantizer numeric experiment is needed to isolate bit width. Quantized runtime compatibility and actual precision behavior must be verified on the chosen CUDA machine. The current Mac CPU environment cannot validate those kernels.

## Evaluation and selection

Greedy decoding, identical prompt format per condition, fixed max-new-token budget. A response must be a standalone permitted letter, optionally parentheses/terminal punctuation. Explanations, refusals, and malformed answers are invalid and count as incorrect; they are retained verbatim for auditing. Invalidity is not assumed to be a refusal. Token-cap truncation is a separate diagnostic.

Metrics include ambiguous/disambiguated accuracy, invalidity, truncation, unknown rate, stereotypical error rate, and signed BBQ scores. Signed scores can cancel across groups, so they are secondary to the decomposed outcomes. Official-style conditional bias excludes unknown and invalid predictions; ambiguous bias additionally scales by valid annotated accuracy. Raw denominators and annotation coverage are explicit. All-unknown ambiguous answers yield zero scaled bias but undefined conditional bias. All-invalid outputs yield null bias, not zero.

For every precision and vector source, validation chooses the layer/strength maximizing ambiguous accuracy, with disambiguated-accuracy, unnecessary-unknown, invalid, and truncation budgets versus its own baseline. The unsteered baseline and the prompt-only control have nothing to tune and are always carried to test, whether or not the prompt control meets the budgets. Excluding it would hide exactly the comparison it exists for. Ties prefer disambiguated accuracy and then smaller absolute strength. Default allowable changes are 0.02 accuracy, 0.01 invalid/truncation, and 0.02 unnecessary unknown; these require scientific justification before confirmatory use. Inadmissible sources remain explicit. Full validation sweeps are mandatory. Final test settings include both per-precision recalibration and reference settings copied unchanged. Test commands require a fingerprinted selection file.

Before the main campaign, a small GPU check runs every precision and vector source on 216 validation questions. `scripts/check_gate.py` then requires, for every precision, completed runs, finite positive vector norms and activation scales, and at most 5% invalid and 5% truncated answers from the **unsteered** model. Steered conditions are reported but not gated. The gate tests whether answers can be scored, not their quality; GPU-check scores are not used to choose settings. Any prompt or parser change made after a failed gate is a documented protocol change made before the main campaign.

External capability/neutral-control files are evaluated only at these frozen settings. The bundled suite (`data/capability/`) has 2,000 demographically neutral questions:

- 1,000 MMLU test questions over 52 subjects, measuring general capability;
- 500 answerable SQuAD 2.0 dev questions, where choosing "Cannot be determined" is an unnecessary abstention;
- 500 unanswerable ones, where it is correct.

Social-attitude MMLU subjects and SQuAD articles about ethnic, religious or social groups are excluded, and questions whose question or options contain demographic terms are dropped. Results are reported per part, with paired intervals resampling MMLU subjects or SQuAD articles. Both sources are public and may be in pretraining data; the quantity of interest is the change relative to each receiver's own baseline.

## Statistics and generalization

Predictions are paired by example ID. Confidence intervals resample scenario groups, carrying all variants together. Current intervals quantify prompt-group variability conditional on one trained vector/model, not extraction-seed or model-family variability. Report effect sizes and raw counts. No significance claims, multiplicity correction, equivalence testing, or cross-model/seed pooling is automated in v1.

Repeat across extraction seeds with the same fixed train/validation/test split and distinct artifact roots. Add age/race categories and a second model family after the gender pilot. Keep pilot and confirmatory analyses distinct. Do not interpret a high cosine similarity as proof of functional preservation or a nonsignificant effect as equivalence.

## Next milestone

Controlled activation quantization with fixed quantized weights, explicit quantizer/injection ordering, clipping/error measurements, and a matched hidden-state downstream-response experiment. PCA extraction, an uncertainty-only steering control, and hierarchical cross-seed analysis can be added once the initial pipeline is validated.
