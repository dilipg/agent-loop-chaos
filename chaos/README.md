# chaos/

    alc run chaos/quickstart.yaml --judge rules

Edit `entrypoint` and the tool name in `quickstart.yaml` to match your agent. Every
failure writes an `AGENT_TASK.md` under `.chaos/` — hand it to a coding agent as-is.
`alc list-faults` shows what else you can inject.
