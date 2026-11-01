# AGENTS.md

Instructions for AI coding assistants working in this repo.

## Read first

- [README.md](README.md): what the project is, where it stands, and the research projects.
- [DETAILS.md](DETAILS.md): how every script works, commands, coordinates, measured results, game rules and the Lovat report format.

## Rules

- **Money:** every Claude API call is billed to the team. Before any run that calls an API, tell the user the estimated cost (DETAILS.md has cost per frame) and wait for a yes.
- **Secrets:** the API key is `CLAUDE_API_KEY` in `.env`. Never print it, log it, put it in a file or commit it.
- **Never commit** `.env`, `.venv/` or `*.mp4`.
- **Answer key:** never overwrite `track/final/`. Write each new run to a new folder, score it with `compare_runs.py`, and add a row to the results table in DETAILS.md.
- **Long jobs:** run them in the background, and pipe output through `tee` to a log so progress shows. Print a progress line at least every 30 s.
- **Python:** use `.venv/bin/python`. Setup is in DETAILS.md.
- **Show results people can check:** labeled videos, tables, and scores against the answer key.
- **Git:** work on a branch, and open a pull request. Keep README.md and DETAILS.md up to date when you change how things work.
