#!/usr/bin/env bash
set -euo pipefail

volume="${N8N_VOLUME:-n8n_data}"
max_age_hours="${N8N_EXECUTIONS_MAX_AGE_HOURS:-24}"
max_count="${N8N_EXECUTIONS_MAX_COUNT:-2000}"

case "$max_age_hours:$max_count" in
  *[!0-9:]*|:*) echo "Retention values must be integers" >&2; exit 2 ;;
esac

# SQLite is installed only in this disposable container; no host package or
# database schema migration is introduced.  Running/waiting executions remain.
docker run --rm -e "max_age_hours=$max_age_hours" -e "max_count=$max_count" -v "$volume:/data" alpine:3.20 sh -ec '
  apk add --no-cache sqlite >/dev/null
  test -f /data/database.sqlite
  before=$(stat -c %s /data/database.sqlite)
  sqlite3 /data/database.sqlite <<SQL
PRAGMA foreign_keys = ON;
CREATE TEMP TABLE doomed_execution_ids (id INTEGER PRIMARY KEY);
INSERT INTO doomed_execution_ids (id)
SELECT id
FROM execution_entity
WHERE status NOT IN (char(110,101,119), char(114,117,110,110,105,110,103), char(119,97,105,116,105,110,103))
  AND COALESCE(stoppedAt, startedAt) < datetime(char(110,111,119), char(45) || ${max_age_hours} || char(32,104,111,117,114,115));
INSERT OR IGNORE INTO doomed_execution_ids (id)
SELECT id
FROM (
  SELECT id
  FROM execution_entity
  WHERE status NOT IN (char(110,101,119), char(114,117,110,110,105,110,103), char(119,97,105,116,105,110,103))
  ORDER BY COALESCE(stoppedAt, startedAt) DESC
  LIMIT -1 OFFSET ${max_count}
);
DELETE FROM execution_data WHERE executionId IN (SELECT id FROM doomed_execution_ids);
DELETE FROM execution_entity WHERE id IN (SELECT id FROM doomed_execution_ids);
VACUUM;
SQL
  after=$(stat -c %s /data/database.sqlite)
  printf "database_bytes_before=%s database_bytes_after=%s retained_execution_limit=%s max_age_hours=%s\\n" "$before" "$after" "$max_count" "$max_age_hours"
'
