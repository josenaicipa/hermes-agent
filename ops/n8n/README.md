# n8n Hostinger migration runtime

This directory runs the recovered n8n 2.34.6 data locally from the Docker
named volume `n8n_data`. It intentionally has no host bind mount for n8n data.
The service listens only on `127.0.0.1:5678`.

## Recovery sequence

1. Create and populate the named volume without writing plaintext data to the
   host:

   ```bash
   ./scripts/restore.sh
   ```

2. Prune completed historical execution data before the first application
   start. The default keeps the last 168 hours and at most 20,000 completed
   executions, while preserving running and waiting executions. The recovered
   database measured 4.72 GB at 168 h/20,000, so the checked-in runtime uses a
   measured 24 h/2,000 retention target to stay near 1.2 GB:

   ```bash
   ./scripts/prune-executions.sh
   ```

3. Start n8n. `up.sh` reads the encryption key directly from 1Password only
   into its child compose process; it never writes an `.env` file or prints the
   key:

   ```bash
   ./scripts/up.sh
   ```

4. Verify the local endpoint and container health:

   ```bash
   curl --fail http://127.0.0.1:5678/healthz
   docker ps --filter name='^/n8n$'
   ```

The public hostname remains `n8n.unlockedacademy.co`. Its Cloudflare Tunnel
ingress is managed outside this repository at
`/mnt/data2tb/services/cloudflared/config.yml`. The file is intentionally
outside this worktree and must be changed only through the existing tunnel
mechanism after local verification. Insert this exact block immediately before
the terminal `- service: http_status:404` entry, then reload only that tunnel:

```yaml
  - hostname: n8n.unlockedacademy.co
    service: http://127.0.0.1:5678
```

Do not add a second proxy or expose port 5678 publicly.

## Rollback

Keep the Hostinger container intact. If public verification fails, restore the
previous DNS record and start the stopped Hostinger n8n container. The local
container can then be stopped with:

```bash
docker compose down
```
