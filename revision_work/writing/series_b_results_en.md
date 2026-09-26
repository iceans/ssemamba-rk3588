# Series B results draft — generated from verified tests

The approved series contains 15 independent training configurations; 15 final-epoch tests have passed verification.

Training uses the fixed protocol in series_b/protocol.json. Historical diagnostic values are excluded. Missing cells remain unavailable. Manuscript locations and reviewer acceptance remain TODO.

Across the three predefined seeds, full achieved AP50 84.652% with sample standard deviation 0.343 percentage points, and mean AP50–95 59.597%. See results/tables/stf_repeats.csv.
Across the three predefined seeds, STF-3DConv achieved AP50 85.984% with sample standard deviation 1.115 percentage points, and mean AP50–95 60.742%. See results/tables/stf_repeats.csv.

All six loss perturbations must be reported with the original default row. No test-selected loss weights replace the main result. The two additional component-repair trainings were not approved, so this series alone does not complete the DAUB LFSS×SAAM factorial table.

The valid-model temporal mismatch results, if available, use a common set at every stride. Missing DFAR, order-only and external-data evidence is not inferred from these runs.

The external protocol was frozen for the official NUDT-MIRSDT test split before target-model scoring (2000 keyframes). Source-domain seed0 final-epoch EMA weights and the predefined confidence threshold0.25 are retained. Exact decoded-image overlap was checked against DAUB train and test; transformed or reused background sources cannot be comprehensively ruled out.
On that frozen external split, full achieved AP50 5.759%, AP50–95 1.037%, and 0.04050 false positives per frame at the predefined source threshold.
On that frozen external split, STF-3DConv achieved AP50 8.963%, AP50–95 1.647%, and 0.03200 false positives per frame at the predefined source threshold.
