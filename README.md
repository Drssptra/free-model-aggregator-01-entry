# Free-model aggregator

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier providers, routing each request
to the best free model that can serve it.

```sh
python3 main.py            # serve (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
python3 main.py --list     # print the catalog document and exit 0
```

Python 3.13, standard library only, nothing to install.

## How it works

- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): the ten providers, the
  sealed quality table (rating, context, tools/vision/JSON per model) and
  the 20-request workload manifest.
- Probes every provider's catalog route in the provider's own wire and
  builds one catalog document. `--list` prints exactly the document the
  server registers with the env.
- Plans each request's primary model before serving: requests are assigned
  in regret order (best model vs best model on another provider), rationing
  each provider's expected free-tier budget, so the top models are spent
  where their edge is biggest and the high-weight requests are served
  first.
- Translates between the client's wire and the chosen provider's wire
  (OpenAI / Anthropic / Gemini) in JSON and in SSE, preserving everything
  the provider echoes: system prompt, user text, tool names, `max_tokens`,
  stop sequences, images and JSON mode. A JSON-mode request from an
  OpenAI client only goes to a wire that can carry it.
- On a 402 / 429 / 5xx, fails over to the next best free model that meets
  the request, never retrying a provider that already refused this
  request. A provider that fails twice in a row (no success between) is
  treated as spent for the rest of the trial.
- Polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the workload is done.
  Every choice is a function of the brief, the requests and the providers'
  answers only: no wall clock, no unseeded randomness, so a replay leaves
  the same state export.

## Local practice scores (kit practice trials, manual runner)

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 25 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 31 |

20/20 requests valid in every trial. Quality 99 is the practical ceiling:
the quality bound ignores per-trial budgets, and the best model for 17 of
the 20 requests is a single OpenRouter model whose budget is 10–12 calls.
