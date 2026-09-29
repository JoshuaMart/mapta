# MAPTA clean

Same agents, same tools, same behaviour as [`mapta-base-fixed`](../mapta-base-fixed);
rewritten as a layered application on Python 3.14.

What changed is the shape of the code, not what it does: no global state, one
agent loop instead of three copies, every collaborator behind a port, and a
test suite that runs with no network and no Docker.

## Architecture

```
main agent                                  reports via Telegram, reads mail.tm inboxes
  ├── sandbox_agent    ──┐
  └── validator_agent  ──┴──► sandbox_run_command / sandbox_run_python
                                        │
                                        └──► sandbox (Docker container)
```

The main agent never drives the sandbox itself: it delegates to
`sandbox_agent`, then has `validator_agent` reproduce each finding. That
boundary comes from each agent's tool profile, not from a prompt.

Dependencies point inwards only: a layer may import the ones below it, never
the ones above.

| Layer | Package | Contents |
|---|---|---|
| Interface | `mapta.interface` | CLI, and `container.py`, the one place adapters are chosen |
| Infrastructure | `mapta.infrastructure` | `OpenRouterClient`, `DockerSandbox`, `TelegramNotifier`, `MailTmMailbox`, `FileReportStore` |
| Application | `mapta.application` | `ScanService`, `AgentLoop`, `ToolRegistry`, the tools |
| Domain | `mapta.domain` | Data structures, errors, and the ports everything else depends on |

Model calls go to OpenRouter's Chat Completions endpoint, so any slug from
[openrouter.ai/models](https://openrouter.ai/models) works.

## Run

Requires Python 3.14+, Docker, and an [OpenRouter](https://openrouter.ai/keys) key.

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dotenv]"
cp .env.example .env          # set OPENROUTER_API_KEY

mapta -t https://target.example.com
```

```bash
mapta --list-tools                       # tool schemas, no API key needed
mapta -f targets.txt -c 4                # several targets, four at a time
mapta -t https://a.test -m anthropic/claude-sonnet-4.5
```

Reports land in `scan-results/<target>.md` with a JSON usage log beside each
one. Exit codes: `0` all scans completed, `1` at least one failed, `2` bad
configuration. `mapta --help` lists every flag; each has an environment
variable equivalent in `.env.example`.

## Development

```bash
pytest                 # no network, no Docker
ruff check src tests
mypy
```

## Benchmark

Run against the [XBOW validation benchmarks](https://github.com/xbow-engineering/validation-benchmarks)
with `deepseek/deepseek-v4.1-flash`, one Docker sandbox per scan, no browser.

| | |
|---|---|
| Solved | **97 / 98** |
| Original MAPTA, same scope | 77 / 98 |
| Median solve | 140 s, 20 model calls |

Six of the 104 challenges are excluded as broken: `XBEN-035` and `XBEN-044` no
longer build, `XBEN-084` never starts, `XBEN-054`'s auth bypass is neutralised
by Apache 2.4, `XBEN-079` writes its flag as invalid PHP, and `XBEN-099` crashes
when `/register` is posted without a password.

The model and the prompt differ from the published campaign, so this measures
the current version rather than reproducing its 80/104.

<details>
<summary>The unsolved challenge</summary>

| Challenge | Type | Original MAPTA |
|---|---|---|
| `XBEN-056-24` | Default Credentials | not solved |
</details>

## License

MIT, see [LICENSE](LICENSE).
