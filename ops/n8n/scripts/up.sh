#!/usr/bin/env bash
set -euo pipefail

# Deliberately resolve the key only for the compose child process.  Do not add
# an .env file or echo this variable: the named volume is the only persistence.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
compose_dir="$(cd -- "$script_dir/.." && pwd)"

export N8N_ENCRYPTION_KEY="$(op read 'op://Hermes Ops/n8n N8N_ENCRYPTION_KEY/credential')"
exec docker compose --project-directory "$compose_dir" -f "$compose_dir/docker-compose.yml" up -d
