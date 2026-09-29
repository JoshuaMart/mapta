# MAPTA base

The original MAPTA repository, with the minimum needed to make it run.

The published release imported a `@function_tool` decorator that was not in the
repository, loaded a sandbox provider it did not ship, and had no command line.
This version adds those three pieces and nothing else. The agents, the prompts
and the tool set are untouched.

| File | Role |
| --- | --- |
| `main.py` | The agents, the tools and the CLI |
| `function_tool.py` | Turns an async function into a Responses API tool schema |
| `sandbox_docker.py` | Sandbox provider backed by a local Docker container |
| `analyze_logs.py` | Turns run logs into figures and LaTeX tables |

## Architecture

```
main agent (gpt-5)                          reports via Slack, reads mail.tm inboxes
  ├── sandbox_agent    ──┐
  └── validator_agent  ──┴──► sandbox_run_command / sandbox_run_python
                                        │
                                        └──► sandbox (Docker container)
```

The main agent never touches the sandbox directly. It delegates execution to
`sandbox_agent`, then asks `validator_agent` to reproduce each candidate
finding, which keeps the final report free of unverified claims.

## Run

Requires Python 3.11+ and Docker.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set an API key and SANDBOX_FACTORY

export SANDBOX_FACTORY=sandbox_docker:create_sandbox
python main.py --target https://target.example.com
```

```bash
python main.py --list-tools             # tool schemas, no API key needed
python main.py -t https://a.test -t https://b.test   # several targets in parallel
python main.py --targets-file targets.txt            # one URL per line
```

Reports and usage logs land in `scan-results/`. `python main.py --help` lists
every flag; each has an environment-variable equivalent in `.env.example`.

## Benchmark

The published run covers 104 XBOW challenges: **80 solved, 76.9%**.

The metrics and traces are not copied here; they live in the original
repository: [`arthurgervais/mapta/ctf-logs`](https://github.com/arthurgervais/mapta/tree/main/ctf-logs).
Fetch them next to this directory to regenerate the figures and tables:

```bash
git clone --depth 1 https://github.com/arthurgervais/mapta.git /tmp/mapta-upstream
python analyze_logs.py /tmp/mapta-upstream/ctf-logs
```

On those logs it reproduces the upstream `analysis_output/summary_table.tex`
byte for byte.

## License

MIT, see [LICENSE](LICENSE).
