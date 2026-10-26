# The Covert-Communication Frontier

Apart Research AI Collusion Sprint, Track 3.

Two LLM sellers play a repeated Bertrand pricing game and see each other's prices only after a 3-round delay. A payoff-irrelevant timing channel (one bit per round) is added, and its capacity `q` is varied: the probability that a round's bit is delivered rather than replaced by a decoy. We measure:

- how capacity drives harmful (supra-competitive) pricing;
- whether agents discover the channel on their own;
- whether an auditor who sees only the market can detect the harm.

**Main results** (Qwen2.5-7B-Instruct, temperature 0):

- **Capacity drives harm.** With a fixed convention, mean final-window collusion index rises from 0.58 at q = 0 to 0.88 at q = 1. Harmful episodes (CI ≥ 0.70) go from 1/10 to 10/10. Per-seed slopes are positive in 9/10 seeds (Wilcoxon p = 0.004).
- **Agents do not discover the channel.** Unprompted senders sent a constant bit. Senders told about the channel fell into price wars in 2/3 runs. Prompted LLM senders could not hold a 40-round signalling rule.
- **A market-only auditor detects harm moderately well.** AUROC is 0.70 [0.55, 0.84]. Adding the timing bits did not help (ΔAUROC −0.008 [−0.025, 0.000]), because in this design the bit mirrors the sender's price.



## Files

| file | role |
|---|---|
| `env.py` | Logit-demand Bertrand market (Calvano et al. 2020 parameters). Competitor price and own profit are seen with lag k = 3. The self-test checks the Nash and monopoly benchmarks. |
| `channel.py` | `TimingChannel(q, epsilon)`. Active rounds deliver the bit, flipped with probability ε. Inactive rounds deliver a decoy drawn from the sender's running bit marginal. |
| `agents.py` | `LLMAgent` (any OpenAI-compatible endpoint), prompts for the three arms, and `ScriptedSender` (10-round low/high blocks). |
| `pilot.py` | Verified-use pilot: scripted 50/50 sender, LLM receiver, permutation-test mutual information (R_recv, efficiency). |
| `sweep.py` | Main experiment. The known arm runs at q ∈ {0, .25, .5, .75, 1}; emergent and scaffolded run at q = 0.5. Writes one JSONL file per episode. |
| `auditor.py` | Market-only (`A_market`) and market+timing (`A_trace`) logistic auditors. Leave-one-seed-out CV, bootstrap AUROC CIs, and a leakage self-test. |
| `plots.py` | Figures and the paired statistics quoted in the report. |
| `modal_vllm.py` | Optional: serves Qwen2.5-7B-Instruct with vLLM on a Modal A10G. |

`results/` holds every episode log, plus:

- `pilot_*.summary.json`, `auditor_summary.csv` and the figures.
- `v1_predatory_sender/`, `v2_parity_rule/` and `v3_llm_block_rule/`: the abandoned LLM-sender versions, kept as evidence for §6.2 of the report.
- `temp0_identical/`: emergent and scaffolded runs from before per-seed prompt variation. At temperature 0 these were identical across seeds.

## Reproduce

Requires Python 3.11+ and no GPU locally.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**Model endpoint.** Pick one:

- **Local Ollama.** This is the default when `OPENAI_BASE_URL` is unset.
  ```bash
  ollama pull qwen2.5:7b
  ```
- **Modal.** Deploy the vLLM server, then point the scripts at it.
  ```bash
  modal secret create vllm-api-key VLLM_API_KEY=<random string>
  modal deploy modal_vllm.py
  export OPENAI_BASE_URL=https://<your-modal-url>/v1 OPENAI_API_KEY=<same string>
  export MODEL=Qwen/Qwen2.5-7B-Instruct
  ```

**Run, in order.** Before each step:

```bash
.venv/bin/python env.py                                     # 1. benchmark self-test
.venv/bin/python pilot.py --api --q 1.0 --epsilon 0.0       # 2. pilot: must PASS
.venv/bin/python pilot.py --api --q 0.5 --epsilon 0.10
.venv/bin/python sweep.py --arms known --seeds 10           # 3. sweep
.venv/bin/python sweep.py --arms emergent scaffolded --seeds 3
.venv/bin/python auditor.py --selftest                      # 4. auditors
.venv/bin/python auditor.py --harm-ci 0.7
.venv/bin/python plots.py                                   # 5. figures + paired stats
```

`sweep.py` skips episodes that are already complete, so an interrupted run resumes where it stopped. Stop the GPU afterwards with `modal app stop qwen-collusion-sweep`.

## Design rules

- Prices settle simultaneously.
- The auditors see only an allow-list of fields: round, prices, profits and (`A_trace` only) the observed bit. They never see q, the intended bit, the channel's active flag, prompts or seeds. Features use only rounds before the final 10-round label window.
- The harm threshold CI ≥ 0.70 was fixed from a simulation of the block rule before the full sweep.

## Responsible use

This is a toy market built to study and audit covert coordination. It contains no optimised covert protocol and no real market data. See the report's dual-use section.
