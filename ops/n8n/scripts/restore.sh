#!/usr/bin/env bash
set -euo pipefail

backup_dir="${1:-/home/jose-naicipa/backups/n8n-hostinger-2026-09-18}"
archive="$backup_dir/rescue-bundle.tar.gz"
volume="${N8N_VOLUME:-n8n_data}"

if [[ ! -r "$archive" ]]; then
  echo "Backup archive is not readable: $archive" >&2
  exit 1
fi

docker volume inspect "$volume" >/dev/null 2>&1 || docker volume create "$volume" >/dev/null

# Restore takes place wholly inside the named volume.  The backup is mounted
# read-only and the host never receives a plaintext SQLite copy.
docker run --rm \
  -v "$volume:/data" \
  -v "$backup_dir:/backup:ro" \
  alpine:3.20 \
  sh -ec '
    test -f /backup/rescue-bundle.tar.gz
    if find /data -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
      echo "Refusing to overwrite a non-empty named volume" >&2
      exit 1
    fi
    tar -xOzf /backup/rescue-bundle.tar.gz ./database.sqlite.gz | gzip -dc > /data/database.sqlite
    tar -xOzf /backup/rescue-bundle.tar.gz ./small/n8n-config > /data/config
    chown 1000:1000 /data /data/database.sqlite /data/config
    chmod 600 /data/database.sqlite /data/config
    printf "restored database_bytes=%s config_bytes=%s\\n" "$(stat -c %s /data/database.sqlite)" "$(stat -c %s /data/config)"
  '
