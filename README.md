# hermes-warm-compaction

A context engine plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). At each compaction, the main model writes the handoff summary on the cached prefix of its last request, so the server reads only the new rows.

The plugin code comes in the first pull request.

## License and credit

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). A copy, a port, or a derivative work of this code must keep the NOTICE file. If you use this design in Hermes Agent or in a different project, please credit Elevatormusic and link to this repository.
