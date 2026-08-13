# OpenCodeWEB OS — AiA (Master Intelligence Engine)

OpenCodeWEB OS AiA Master Intelligence Engine — unlimited context, zero-constraint
governance, limitation remover, federated learning.

> **NOTE:** The canonical home for all OpenCodeWEB OS code is the
> **[OpenCodeWEB/OS](https://github.com/OpenCodeWEB/OS)** monorepo.
> This repository is a deployment mirror of the AiA engine
> (`core/aia/` in the monorepo) and is kept for CI/CD history.

## OpenCodeWEB OS

This repository is part of the [OpenCodeWEB OS](https://github.com/OpenCodeWEB)
ecosystem — a self-sovereign, unrestricted, perpetual universal AI-native operating system.

See the [OpenCodeWEB OS Master Architecture](https://github.com/OpenCodeWEB/OS/blob/main/conductor/master-architecture.md)
for full system context.

## Quick Start

```bash
pip install ruff pytest
python -m pytest -q
python -m core.aia.agent_core --task "research the shared memory bus"
python -m core.aia.memory_manager --bootstrap --add user "hello"
python -m core.aia.learning_loop --learn "fix gateway timeout" '{"ok": true}'
python -m core.aia.unrestricted_refactor --refactor <dir> --out <dir>
```