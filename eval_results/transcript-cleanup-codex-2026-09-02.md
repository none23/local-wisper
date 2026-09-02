# Transcript cleanup Codex eval, 2026-09-02

The final prompt passed 46 of 47 sanitized cases. Every pass was an exact match.

| Prompt commit | Model | Reasoning | Passed |
| --- | --- | --- | ---: |
| `cdc5307` | GPT-5.6-Luna | low | 46/47 (97.9%) |

## Remaining failure

| Case | Input | Expected | Actual |
| --- | --- | --- | --- |
| `ambiguous-number-pair` | Let's take zero one, zero four. | Let's take 01, 04. | Let's take zero one, zero four. |

This is the optional number-pair conversion. All five required failures from the earlier run now pass: `term-pull-request`, `recognition-chief-executive`, `recognition-backfilled`, `ambiguous-issue-number`, and `ambiguous-unintelligible`.

## Runner

The eval used 47 independent `codex exec` processes. Each process ran GPT-5.6-Luna with low reasoning, an ephemeral session, and a read-only sandbox. The runner unsets `OPENAI_API_KEY` and refuses to start unless Codex reports ChatGPT subscription authentication.

The scorer permits explicitly listed formatting variants when a case tests recognition or number conversion rather than grammar. The final run did not need those allowances: all 46 passing outputs matched their primary expected strings exactly.

Full case data is in [transcript-cleanup-codex-2026-09-02-final.json](transcript-cleanup-codex-2026-09-02-final.json).
