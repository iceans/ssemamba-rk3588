# Manuscript revision snippets — draft, not applied to a manuscript

The submitted tex/PDF/bib was not found. The following English text is ready for editorial use with the stated limitations. Equations follow the audited current implementation; the final experimental version must be frozen before insertion. No manuscript was compiled.

## Contribution and claim scope

> We investigate shallow spatiotemporal fusion for infrared video small-target detection. The framework combines a temporal stem with feature fusion at high spatial resolution and a keyframe detection head. Our contribution concerns this specific integration and its controlled evaluation. We do not claim that state-space models are indispensable, that interleaving is itself new, or that adjacent sequence positions guarantee preservation of small targets. Alignment and task-oriented attention are auxiliary implementation components; their effects must be assessed within the defined ablations.

## SFIScan definition and computational path

Let \(m,s\in\mathbb{R}^{C\times H\times W}\), \(L=HW\), and let \(m_i,s_i\) denote features at raster index \(i\). The audited two-stream implementation constructs

\[
u_{2i}=m_i,\qquad u_{2i+1}=s_i,\qquad 0\leq i<L.
\]

For \(T\) feature streams, it uses \(u_{Ti+t}=x_i^{(t)}\). The selective scan receives input-dependent \(\Delta_k,B_k,C_k\) and the learned \(A=-\exp(A_{\log})\), with positive steps obtained by softplus. Suppressing batch/channel indices, its state computation can be written as

\[
\bar A_k=\exp(\Delta_k A),\qquad
h_k=\bar A_k\odot h_{k-1}+\Delta_k B_k u_k,\qquad
v_k=C_k h_k+D u_k,\quad h_{-1}=0.
\]

The two-stream output is reconstructed as \(z_i=v_{2i}+v_{2i+1}\); the multistream version uses \(z_i=\sum_{t=0}^{T-1}v_{Ti+t}\). Thus the current implementation uses a sum. Normalization, projection and residual/MLP operations then follow the enclosing block. Its active fusion route uses raster order; the availability of multidirectional scan helpers does not imply that those routes are active here.

For a controlled blocked-order comparison, the same streams and length \(2L\) would instead use \(u_i=m_i,u_{L+i}=s_i\), with reconstruction \(z_i=v_i+v_{L+i}\). These index rules coincide under an identity replacement after the same normalization, but the learned causal recurrences need not produce equal values. Interleaving changes the computational path and contextual order; it does not establish a detection guarantee. The currently named wosfi branch removes a stream and is therefore excluded from a pure order-only interpretation.

Implementation references: `timeguid.py` SS2DguideMix, `tgsmamba.py` SS2D1conv and selective_scan_torch. This is a computational description, not a performance theorem or a claim of exact continuous-time integration.

## STF replacement boundary

> The STF-3DConv variant replaces the temporal stem as a block, including its local depthwise operations, alignment and temporal state-space computation. State-space modules in IFSS and LFSS remain. This comparison therefore concerns two temporal-stem designs and cannot isolate the contribution of the SSM operator alone. A valid comparison must additionally hold the neck routing, channels, detection head and training protocol fixed.

## Related-work bridge

> Temporal interleaving has precedent in MOCID's displacement-aware Mamba, while ViViM explores spatial and temporal scan orderings for video segmentation. These precedents motivate a comparison of the concrete fusion and reconstruction design rather than a claim of novelty for sequence interleaving alone. [MOCID](https://ojs.aaai.org/index.php/AAAI/article/download/33087/35242), [ViViM](https://arxiv.org/html/2401.14168v3).

Use `technical_comparison.md` for the six-work comparison; citation keys should be reconciled against the actual bibliography rather than guessed.

## Evidence-limited dataset description

> Annotation statistics refer to the maximum bounding-box side in original-image pixels, not the physical emitting target size. In the audited local DAUB held-out directory, all 4,795 annotations exceed 7 pixels; the ≤3 and (3,7] bins contain no annotations. The audited IRDST validation split contains 618 annotations in (3,7] and 20,145 above 7 pixels, with none at ≤3 pixels. These data do not establish coverage of single-pixel targets. Scene-specific claims require independently verified sequence metadata.

## Results passages requiring formal evidence

TODO: insert each model/seed result, paired differences, means and sample standard deviations (ddof=1) only after TEST tasks are verified. Do not write a standard deviation for one run. TODO: complete DAUB LFSS×SAAM four combinations, kernel 1/3/5 and all six loss perturbations; retain unfavorable planned results. IRDST must retain an incremental table if E1 is absent.

TODO: P1 must report 4,683 common DAUB keyframes for every stride, including the re-evaluated stride-1 point, and call the test temporal sampling mismatch. TODO: P2 needs an ignore-aware size evaluator and a source-fixed false-alarm threshold. TODO: P3 needs a single frozen external dataset/split and common DAUB-source weights. None of these results is claimed completed.

## Reproducibility note for the working record

The completed diagnostic re-evaluations must remain separate from manuscript test results: AP50/AP50–95 are 87.588/58.030% for the fullsize candidate and 94.868/63.066% for the STF-3DConv candidate. Their historical checkpoint selection used the currently configured test directory as validation, and the current loader duplicates/omits keyframes. These numbers cannot establish a controlled improvement or reverse the model ranking in a formal table. The full-model discrepancy remains unresolved.

Suggested manuscript locations: contribution paragraph in Introduction; scan definition and STF boundary in Method; annotation coverage and protocol in Experimental Settings; evidence limitations in Discussion. All final page/line references are TODO.
