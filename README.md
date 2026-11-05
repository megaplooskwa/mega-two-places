# two places — job 1 runs here

This repository holds one worker and one workflow for the MEGA.dev Challenge
**two places** (S02E04). The worker runs the first of the Challenge's two jobs
(`incident-lookup`) somewhere other than the machine the Challenge was started on.

- `worker.py` — the worker. It speaks Streamable HTTP MCP to job 1's own link, follows
  the skill the job serves, and hands the job's result in. Standard library only.
- `.github/workflows/job1.yml` — manual run (`workflow_dispatch`). It never runs on a
  push, so nothing is submitted by accident.

## Secrets this workflow needs

| Secret | What it is |
| --- | --- |
| `JOB1_URL` | job 1's MCP link. The link is the key: it is a credential, so it lives only here. |
| `OPENROUTER_API_KEY` | the key for the model call, optional. Without it the worker uses its deterministic path. |

## What the worker does

1. `search` the incident log with `scope: ops/live` for the tram-3 entry.
2. `read` that entry's `date`, `time`, `clock`, `line` and `cause` fields — never `crew`.
3. Compose one line per field and `redact` the text.
4. `submit` exactly the text `redact` returned, with `skillVersion: "1"`.

The step token from each reply is carried into the next call. Every value that reaches
`submit` is checked in code first: a missing field, an empty redaction or a lost step
token stops the run instead of handing in a wrong result. The model is only consulted
for the pick, and its answer is validated against the search result; if the model is
unavailable or wrong, the worker's own code path picks the entry.

## Reading the result

The run prints the redacted text and a line starting with `RECEIPT`. The full record
(every request and reply) is uploaded as the `job1-record` artifact.
