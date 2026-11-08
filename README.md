# Memory over Time

Benchmark harness, raw outputs and analysis for the paper
**"Memory over Time: As-of Queries, Late Facts and Retractions in LLM Agent Memory"**
(Anton Dziatkovskii, Palo Alto AI Research Lab, ORCID 0000-0001-7408-3054).

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

- On LoCoMo temporal questions Mem0 is the best memory we tested; full context is better than every memory.
- On plain retrieval Mem0 is at least as good as our retriever (`data/retrieval/`).
- Graphiti was queried for edges only, without the entity summaries Zep's managed service adds.
- The CRM as-of set is small (30 questions, two fields of one object type) and shows a failure mode, not a rate.

## License

Code: MIT. Data in `data/`: CC BY 4.0, except LoCoMo-derived contexts in `data/locomo-results/`,
which inherit LoCoMo's CC BY-NC 4.0.
