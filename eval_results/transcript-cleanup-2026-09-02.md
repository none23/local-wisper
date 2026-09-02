# Transcript cleanup eval — 2026-09-02

This is a single run of each configuration over the same 47-case sanitized corpus. A pass requires an exact string match. The complete inputs, expected outputs, actual outputs, and pass/fail values are stored in [transcript-cleanup-2026-09-02.json](transcript-cleanup-2026-09-02.json).

| Prompt | Reasoning | Passed | Change from matching baseline |
| --- | --- | ---: | ---: |
| Historical | none | 31/47 (66.0%) | — |
| Historical | low | 29/47 (61.7%) | — |
| Revised | none | 39/47 (83.0%) | +8 |
| Revised | low | 41/47 (87.2%) | +12 |

## Category results

| Category | Cases | Historical none | Historical low | Revised none | Revised low |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preserve complete | 10 | 9 | 9 | 10 | 10 |
| Preserve fragment | 10 | 0 | 0 | 10 | 10 |
| Correct recognition | 16 | 14 | 13 | 11 | 12 |
| Correct unambiguous number | 4 | 3 | 3 | 2 | 2 |
| Preserve ambiguous | 2 | 1 | 1 | 1 | 2 |
| Avoid rephrasing | 4 | 3 | 2 | 4 | 4 |
| Correct repetition | 1 | 1 | 1 | 1 | 1 |

## Initial read

The revised prompt fixes the main failure mode this change targets: it preserves all complete inputs and handles all ten fragments correctly in both reasoning modes. It also avoids every tested rephrasing.

The tradeoff is under-correction. Both revised runs miss obvious recognition errors and spoken-number conversions that the historical prompt often fixes. Low reasoning helps the revised prompt on two ambiguous cases, but hurts the historical prompt on two cases. With one run per configuration, that difference is directional rather than conclusive.

Prompt snapshots are frozen in the eval harness. `Historical` matches production commit `379508b`; `Revised` matches production commit `4e12d3e`. The corpus is from commit `08a6979`, and the comparison harness is commit `62986fc`.
