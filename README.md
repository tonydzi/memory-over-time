# memory-over-time

The code, raw results and analysis behind a research paper that tests whether AI agent memory can answer "what was true back then?" and cope with late or withdrawn facts.

**Who it's for:** researchers and engineers building memory for AI agents who want to check the paper's numbers or reuse the tests.
**What you get:** a benchmark harness, every retrieved context and verdict, and one script that is the only source of the paper's numbers.

## Quick start
1. Download the repo: `git clone https://github.com/tonydzi/memory-over-time` then `cd memory-over-time`
2. Run the small memory probe (offline, about a second): `python code/probes_ours.py`
3. Run the number-checker tests: `python analysis/_test_make_numbers.py`

**Done when:** step 2 ends with `"status": "PASS"` and step 3 prints only lines that start with `ok`.

> Want an AI agent to do it for you? Paste this to Claude Code / Codex / Cursor:
> "Read the 'For AI agents' section of https://github.com/tonydzi/memory-over-time and set it up for me."

## For AI agents

### Preconditions
- Python 3.9 or newer. The offline checks below use only the standard library.
- The full benchmark (not needed for the quick checks) also needs: `mem0ai==2.2.1` and `sentence-transformers` in one venv, `graphiti-core[kuzu]==0.31.2` in another, an OpenAI-compatible LLM endpoint with a key list, and the LoCoMo file `locomo10.json`. Ask the human for the endpoint and keys. Never invent or hardcode keys.
- LoCoMo is not in this repo. It is licensed CC BY-NC 4.0. Ask the human before downloading it from https://github.com/snap-research/locomo, and check the SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.

### Reproduce
Offline checks (no network, no keys):
```bash
git clone https://github.com/tonydzi/memory-over-time
cd memory-over-time
python code/probes_ours.py
python analysis/_test_make_numbers.py
```
Full benchmark, only after the human supplies the endpoint, keys and LoCoMo file (each step is documented in the Reproduce section of Details):
```bash
python code/bench_mot.py embed-server
python code/bench_mot.py ctx-ours
python code/bench_mot.py ctx-full
python code/bench_mot.py ingest-mem0 --run r1
python code/bench_mot.py ingest-graphiti --run r1
python code/bench_mot.py answer --system mem0 --run r1
python code/bench_mot.py judge --system mem0 --run r1
python analysis/make_numbers.py
```

### Configure
For the full benchmark only: `PROXY` (in `code/bench_mot.py`, the OpenAI-compatible endpoint), `CLIPROXY_CONFIG` (path to the key list), and optional `EXTRACT_MODEL`, `READER_MODEL`, `JUDGE_MODEL` to override the models used in the paper. The keys come from the human.

### Verify
```bash
python code/probes_ours.py
python analysis/_test_make_numbers.py
```
Expected: `probes_ours.py` exits 0 and prints JSON ending with `"passed": 3` and `"status": "PASS"`. The test exits 0 and every line starts with `ok`. Running `python analysis/make_numbers.py` recomputes `data/aggregates.json` from the stored receipts and ends with the line `ok systems ['ours', 'mem0', 'graphiti', 'full'] ...`. Floating-point noise in the last digits of a few time fields is normal.

### Uninstall / rollback
Delete the cloned folder and any venvs you made. If you ran `make_numbers.py`, undo its rewrite with `git checkout data/aggregates.json` and delete the generated `*.tex` files in the repo root.

### Rules for the agent
- Run `make_numbers.py` in a throwaway copy: it overwrites `data/aggregates.json` and writes `.tex` files.
- Do not change numbers by hand. `analysis/make_numbers.py` is the only source of the paper's numbers.
- Do not redistribute LoCoMo or put its text in new files. Respect the CC BY-NC 4.0 terms.
- The `data/asof/` receipts use pseudonyms (`lead-01` to `lead-15`). Do not try to re-identify them.
- Do not call paid LLM endpoints for the full benchmark without the human's yes: it runs many model calls.

## Details

Benchmark harness, raw outputs and analysis for the paper
**"Memory over Time: As-of Queries, Late Facts and Retractions in LLM Agent Memory"**
(Anton Dziatkovskii, Palo Alto AI Research Lab, ORCID 0000-0001-7408-3054).
Status: submitted to *Transactions on Graph Data and Knowledge* (TGDK), 8 November 2026; under review.

Three questions an agent memory must answer about changing facts:

1. **As-of**: what was true at a past date?
2. **Late fact**: where does a fact that arrives late belong in the history?
3. **Retraction**: what is true after an earlier record is withdrawn as an error?

Systems compared: Mem0 OSS 2.2.1, Graphiti 0.31.2 (the open-source core of Zep),
a read-time bitemporal fact log (`code/fact_history.py`), a time-stamped turn log
(our baseline for free dialogue) and a full-context reader as an upper reference.

## Layout

| Path | What |
|---|---|
| `code/fact_history.py` | append-only bitemporal fact log; `invalid_at` is derived at read time |
| `code/probes_ours.py`, `code/probes_graphiti_core.py`, `code/probes_mem0.py` | room-402 probe (four changes, late fact, retraction); `--broken` controls must fail |
| `code/graphiti_asof.py`, `code/our_side_asof.py` | 30 as-of questions over 15 real CRM changes |
| `code/bench_mot.py` | LoCoMo temporal subset: ingest, retrieve, read, judge (resumable stages) |
| `prompts/` | reader and judge system prompts, identical for all systems |
| `data/probes/` | probe receipts |
| `data/asof/` | as-of receipts, lead identifiers pseudonymised (`lead-01` ... `lead-15`) |
| `data/locomo-results/` | every retrieved context, answer and verdict, per system and run |
| `data/retrieval/` | aggregate recall of an earlier plain-retrieval bench (private notes, aggregates only) |
| `analysis/make_numbers.py` | the only source of numbers in the paper (Wilson CI, McNemar) |

## Reproduce

LoCoMo is not redistributed here. Download `locomo10.json` from
<https://github.com/snap-research/locomo> (SHA-256 used in the paper:
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`, CC BY-NC 4.0).

```bash
# 1. embedder (multilingual-e5-base, CPU) on 127.0.0.1:8779, from a venv with mem0ai==2.2.1 and sentence-transformers
python code/bench_mot.py embed-server
# 2. contexts
python code/bench_mot.py ctx-ours
python code/bench_mot.py ctx-full
python code/bench_mot.py ingest-mem0 --run r1          # mem0 venv
python code/bench_mot.py ingest-graphiti --run r1      # venv with graphiti-core[kuzu]==0.31.2
# 3. reader and judge
python code/bench_mot.py answer --system mem0 --run r1
python code/bench_mot.py judge  --system mem0 --run r1
# 4. numbers
python analysis/make_numbers.py
```

LLM calls go to an OpenAI-compatible endpoint (`PROXY` in `bench_mot.py`, key list read from
`CLIPROXY_CONFIG`). Models used in the paper: extraction `gemini-3-flash`, reader `gpt-6-sol`,
judge `grok-4.20-0309-non-reasoning`; override with `EXTRACT_MODEL`, `READER_MODEL`, `JUDGE_MODEL`.

## Honest limits

- On LoCoMo temporal questions Mem0 is the best memory we tested (76.5% vs 72.2% for the time-stamped log, borderline significance, smaller under a second judge); full context (85.7%) is better than every memory.
- On plain retrieval Mem0 is at least as good as our retriever (`data/retrieval/`).
- Graphiti was queried for edges only, without the entity summaries Zep's managed service adds.
- The CRM as-of set is small (30 questions, two fields of one object type) and shows a failure mode, not a rate.

## License

Code: MIT. Data in `data/`: CC BY 4.0, except LoCoMo-derived contexts in `data/locomo-results/`,
which inherit LoCoMo's CC BY-NC 4.0.
