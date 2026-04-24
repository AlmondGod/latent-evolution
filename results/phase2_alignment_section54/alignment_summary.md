# Section 5.4 Alignment Summary

- Paired runs: 3
- ALEC mean Spearman: -0.270 +/- 0.263
- RL mean Spearman: 0.413 +/- 0.190
- Paired Spearman test: exact_sign_flip (p=0.500)
- ALEC top-match rate: 0.0% +/- 0.0%
- RL top-match rate: 0.0% +/- 0.0%
- Paired top-match test: exact McNemar (p=1.000, ALEC-only=0, RL-only=0)

Paper-ready sentence:

The mean within-run Spearman correlation between search fitness and held-out reward was $-0.270 \pm 0.263$ for ALEC versus $0.413 \pm 0.190$ for the matched-budget RL baseline (mean $\pm$ s.e.m.; paired sign-flip test $p=0.500$). The top search-time candidate remained the top held-out candidate in $0.0\% \pm 0.0\%$ versus $0.0\% \pm 0.0\%$ of runs (paired exact McNemar test $p=1.000$).
