# Contributing

Thank you for your help. This guide tells you how to report a problem, propose a change, and run the checks.

## Report a problem

- For a bug, open an issue with the **Bug report** form. Give the Hermes Agent commit, the plugin version, the API mode, and the server (for example vLLM, llama.cpp, or a hosted API).
- Each compaction writes one `warm_compaction:` line to `logs/agent.log`. Add that line: it has the path (`warm`, `fallback`, or `fixed`), the reason, and the time. It has no conversation text.
- Do not attach private conversations, API keys, or full request bodies. Use a synthetic conversation that shows the problem.
- For a security problem, do not open an issue. See [SECURITY.md](SECURITY.md).

## Propose a change

1. Open an issue first for a large change, so that we can agree on the design.
2. Fork the repository and make a branch from `main`.
3. Write a test that fails without your change. Then make the change.
4. Run the checks below.
5. Open a pull request. Fill in the template.

Keep each pull request to one subject. The plugin uses documented Hermes plugin APIs only: no patch of Hermes, no subclass of the built-in compressor, and no runtime wrapping of Hermes code. A change that needs one of those cannot be merged.

## Run the checks

You need Python 3.10 or later and Git. The plugin has no third-party dependency.

```bash
python -m unittest discover -s tests -p "test_wc_*.py"
```

Lint with [ruff](https://docs.astral.sh/ruff/) (the version that CI uses):

```bash
pip install ruff==0.16.10
ruff check .
```

The integration check runs real Hermes code against a loopback fake server. Run it with the Python of a Hermes virtual environment and a **clean** Hermes checkout, not your installed Hermes:

```bash
<hermes-venv-python> -B scripts/check_plugin_hermes.py --hermes-source <clean-hermes-checkout> --report .work/plugin-integration-report.json
```

CI runs the unit tests on Python 3.10 to 3.13 (Linux) and on Python 3.12 (Windows and macOS), and the lint. A pull request must pass both.

## Style

- Match the code around your change: names, comment density, and line length (120).
- Write comments, docstrings, and documents in short, simple sentences.
- A measured result needs its record in `evidence/` with metadata only. Report a missing counter as unknown.

## License

By contributing, you agree that your contribution is licensed under the [Apache License 2.0](LICENSE). Keep the [NOTICE](NOTICE) file in copies and derivative works.
