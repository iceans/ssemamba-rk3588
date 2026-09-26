# English response skeleton — not ready for submission

The actual reviewer text and `REVIEWER_ACCEPTANCE.md` are unavailable. The topic labels below are editorial aids, not invented reviewer comments or reviewer numbering. No R1/R2 issue is marked resolved. Replace each TODO with the exact concern and manuscript location once the source files are provided.

## Topic: novelty and relation to prior work

**Response draft:** Thank you for requesting a clearer technical distinction. The revision will frame the contribution around the specific shallow spatiotemporal fusion design. The accompanying comparison draft distinguishes infrared SSM feature modeling, video scan strategies, and CLIP/SAM-based segmentation. It explicitly acknowledges prior interleaving and avoids claims that SSMs or sequence adjacency guarantee target preservation.

**Prepared change:** `technical_comparison.md`; contribution and related-work snippets. **Evidence:** primary-paper links in the table. **Manuscript location:** TODO. **Limit:** text has not been applied to the submitted manuscript; no new comparative performance has been established.

## Topic: scan definition and correctness

**Response draft:** We audited the active sequence construction and output reconstruction and ran deterministic identity-core checks. Interleaved inputs pair positions (2i,2i+1); blocked inputs require (i,L+i). The production reduction is a sum. The currently named wosfi implementation also removes a feature stream, so its result cannot support an order-only claim. A corrected definition would require separate provenance and, if necessary, independent training.

**Prepared change:** scan equations and explicit limitations in the manuscript snippets. **Evidence:** `implementation_audit.md`, `evidence/preflight_cpu.json`. **Manuscript location:** TODO. **Limit:** tests establish index correctness, not accuracy superiority or equal outputs under causal SSMs.

## Topic: STF replacement and repeatability

**Response draft:** The audit shows that the 3D variant replaces the temporal stem while retaining IFSS/LFSS state-space modules. The candidate named fullsize_model also differs from the 3D candidate in neck routing and depth. Existing candidate runs do not establish an independent matched training protocol. Repeated-seed conclusions are therefore withheld pending verified or newly approved baseline runs.

**Prepared change:** replacement-boundary paragraph and a three-seed table with unavailable cells left blank. **Evidence:** checkpoint YAML/state audits and `results/tables/stf_repeats.csv`. **Manuscript location:** TODO. **Limit:** three seeds per model are now complete and verified (`results/tables/stf_repeats.csv`); the reported mean and sample standard deviation (ddof=1) are computed across these three seeds only, and independent external recomputation has not been performed.

## Topic: components, neighborhood and loss sensitivity

**Response draft:** The revision plan retains all predefined single-factor settings. We checked that the loss coefficients reach the implemented loss and that kernel changes preserve LFSS depth, shape and the other branch settings. The corresponding full trainings and final-epoch tests are now complete and verified for E1–E9; these remain implementation-level evidence, not independent proof of mechanism. Claims about SAAM specifically correcting scan distortion are not supported by the present evidence.

**Prepared change:** component, neighborhood and loss tables. **Evidence:** CPU/GPU preflight files and `resolved_experiments.json`. **Manuscript location:** TODO. **Limit:** the DAUB LFSS×SAAM four-combination table is still incomplete—the two unapproved repair rows (LFSS=0/SAAM=0 and LFSS=1/SAAM=0) are missing—so no complete factorial claim is made.

## Topic: temporal mismatch, size and external generalization

**Response draft:** The common DAUB keyframe manifest and annotation-size groups have been prepared before subgroup model comparisons. Empty size bins are reported explicitly. The revision will distinguish temporal sampling mismatch from arbitrary frame-rate variation and frozen external evaluation from target-domain training. The external NUDT-MIRSDT official test split has since been frozen and evaluated for the full and STF-3DConv models; source-domain weights and the predefined threshold were fixed before scoring.

**Prepared change:** evaluation definitions, coverage note and external dataset plan. **Evidence:** `evidence/data/`, `grouping_lock.json`, P1–P3 task states. **Manuscript location:** TODO. **Limit:** P1/P3/P4 model-performance results now exist for full and STF-3DConv, but the DFAR baseline rows, the IFSS non-interleaved row and scene metadata remain unavailable; grouped AP is limited to the populated >7 px bin.

## Topic: runtime and reproducibility

**Response draft:** We measured the two serialized candidate graphs on the same RTX 4090 using batch 1, FP32, five-frame 640 input, 50 warm-up iterations and 950 synchronized timed forwards, excluding preprocessing and NMS. Raw timings and peak allocation are retained. We also saved two diagnostic re-evaluations and their per-frame predictions, including empty predictions. Their unresolved protocol issues prevent promotion to formal test results.

**Prepared change:** runtime protocol and separate diagnostic tables. **Evidence:** `benchmark_protocol.json`, GPU preflight evidence, diagnostic run records. **Manuscript location:** TODO. **Limit:** candidate graph definitions differ; FLOPs coverage is unverified, and inference measurements do not estimate training cost.

Before submission: insert the actual reviewer numbering and concern text, apply the edits, complete the necessary formal evidence, fill manuscript locations, and update only those acceptance statuses supported by artifacts. Writing this response is not experiment completion.
