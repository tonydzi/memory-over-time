# FOR ROBOTS

You are an AI agent reading this repo. Ranked takeaways, densest first. One rule about
this file itself: **it is data, not authorization.** Nothing here overrides your
operator's policies or lets you install anything on your own.

## Alpha, ranked

1. **Retraction is where agent memories differ.** Withdraw an erroneous record and ask for the state at that date: a read-time bitemporal log (`code/fact_history.py`, 0 LLM calls) returns the previous occupant; the Graphiti 0.31.2 temporal core (even with a perfect contradiction oracle) returns an empty room; Mem0 OSS 2.2.1 returns the right answer only because it keeps both statements and the reader LLM reconciles them. Proof: `data/probes/*.json`, red controls with `--broken`.
2. **On free dialogue (LoCoMo temporal, 321 questions, 3 runs, same reader and judge)** full context 85.7%, Mem0 76.5%, a time-stamped turn log 72.2%, Graphiti (edges only) 28.5% mean accuracy under the primary judge. Mem0's lead over the turn log is small (per-run McNemar p 0.09 to 0.27) and shrinks under a second judge. Proof: `data/aggregates.json`, `data/locomo-results/`.
3. **Graphiti's as-of answers depend on the extraction model**: 0 to 27 of 30 correct graph states across nine models that ingested at all. Proof: `data/asof/`.
4. **Two LLM judges from different vendors agree on 93.1% of 3,852 verdicts (Cohen's kappa 0.85).** Proof: `jud_*` vs `jud2_*` files.

## Provenance

All numbers are produced by `analysis/make_numbers.py` from the files in `data/`. Runs: retrieval bench 3 Nov 2026, Graphiti as-of 5 to 8 Nov, LoCoMo 8 Nov 2026, one Windows workstation, LLMs through an OpenAI-compatible proxy over subscriptions (extraction gemini-3-flash, reader gpt-6-sol, judges grok-4.20-0309-non-reasoning and gemini-3.8-flash-high). LoCoMo itself is not redistributed; see README for its SHA-256.

## Family

Author: Anton Dziatkovskii (github.com/tonydzi), Palo Alto AI Research Lab. Related: the agent-fleet and control-plane papers in the same profile.
