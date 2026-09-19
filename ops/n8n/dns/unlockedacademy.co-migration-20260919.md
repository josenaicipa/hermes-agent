# Fase 3 — Cloudflare Tunnel para `n8n.unlockedacademy.co`

Fecha: 2026-09-19. Estado: detenido antes de cualquier mutación efectiva de
Cloudflare o del host.

## Evidencia y bloqueo

- `http://127.0.0.1:5678/healthz` respondió `200`.
- El túnel configurado en el host es
  `b1ea5480-4905-4522-9c45-529a64b56e66`.
- Antes de crear, `GET /zones?name=unlockedacademy.co` devolvió una lista
  vacía. Tras el intento fallido se volvió a comprobar la misma condición.
- `POST /zones` devolvió HTTP 403 con:
  `Requires permission "com.cloudflare.api.account.zone.create" to create zones for the selected account`.
- Falta exactamente el permiso Cloudflare **Zone:Create**
  (`com.cloudflare.api.account.zone.create`) para la cuenta seleccionada. No
  se creó una zona, no hay nameservers Cloudflare asignados y no se debe
  continuar hasta que un token con ese permiso esté disponible.

El token se obtuvo mediante `op run` desde el item indicado y nunca se guardó
ni se imprimió. No se usó la API de GoDaddy.

## DNS preservado para cuando se desbloquee

El inventario previo está en
`ops/n8n/dns/unlockedacademy.co-godaddy-20260919.txt`. Deben replicarse con
`proxied=false` los registros heredados: A=2 de apex, MX=1, TXT=3 y CNAME=1
(`www`). El A heredado de `n8n` debe sustituirse por el único CNAME:

```text
n8n  CNAME  b1ea5480-4905-4522-9c45-529a64b56e66.cfargotunnel.com  proxied=true
```

No se migran los NS/SOA de GoDaddy: Cloudflare los administra al crear la
zona. El MX de Mailgun es crítico y se replica sin proxy.

## Diff de ingress propuesto (no aplicado)

Archivo objetivo: `/mnt/data2tb/services/cloudflared/config.yml`, justo antes
de la regla final `service: http_status:404`.

```diff
 ingress:
   # ... reglas existentes sin cambio ...
+  - hostname: n8n.unlockedacademy.co
+    service: http://127.0.0.1:5678
   - service: http_status:404
```

La secuencia segura pendiente es: crear/verificar primero
`/mnt/data2tb/services/cloudflared/config.yml.bak-20260919`; aplicar el diff;
leer el archivo y `journalctl -u cloudflared-agentes -n 50` para confirmar la
recarga sin reiniciar la unidad. La allowlist del router rechazó ambos paths
del host, por lo que esta ejecución tampoco tenía cobertura autorizada de
escritura para esos dos archivos.

## Post-corte, solo tras zona activa

1. Comparar programáticamente los registros de Cloudflare contra el
   inventario y corregir únicamente faltantes de esta zona; verificar todas
   las escrituras con lecturas posteriores.
2. Crear el CNAME proxied de `n8n`, configurar SSL `Full` y `Always Use
   HTTPS`, y anotar los dos nameservers asignados.
3. Validar por `--resolve` usando una IP actual de
   `school.unlockedecom.co`; antes de delegar NS, un 4xx de Cloudflare es
   esperado porque la zona no está activa.
4. Fable/Jose cambia los NS en GoDaddy. Esperar zona `active`, comprobar
   `/healthz` y login por el dominio, y ejecutar un webhook solo si hay uno
   demostrablemente inocuo.
5. Detener `n8n-v2` en Hostinger sin borrarlo. Rollback: revertir NS en
   GoDaddy y hacer `docker start n8n-v2`.

No se cambiaron NS de GoDaddy, no se reinició `cloudflared-agentes`, ni se
tocaron otras zonas Cloudflare.

## Continuación Fase 3b — 2026-09-19

Esta continuación revalidó el estado antes de escribir y creó la zona
Cloudflare con `type=full`. La zona queda deliberadamente en `pending` hasta
que el propietario sustituya la delegación en GoDaddy; esta ejecución no llamó
la API de GoDaddy ni tocó el servidor Hostinger `72.61.7.174`.

### Zona y DNS creados

- Zone ID: `31dfec0fa584ae70e4556aee65d79170`.
- Nameservers asignados: `maisie.ns.cloudflare.com` y
  `troy.ns.cloudflare.com`.
- La credencial de `cloudflared-agentes` fue comprobada contra la cuenta de la
  zona y corresponde a la misma cuenta. El túnel usado es
  `b1ea5480-4905-4522-9c45-529a64b56e66`.
- Se crearon 8 registros: los 7 heredados del inventario sin el A legado de
  `n8n`, más el CNAME proxied de `n8n` al túnel.

| Nombre | Tipo | Destino/contenido | TTL | Proxy |
| --- | --- | --- | --- | --- |
| `unlockedacademy.co` | A | `162.159.142.17` | 60 | no |
| `unlockedacademy.co` | A | `172.66.2.13` | 60 | no |
| `unlockedacademy.co` | MX | `10 mxa.us.mailgun.org` | 3600 | no |
| `unlockedacademy.co` | TXT | `lw=68dc04362e776ced248cac8e` | 3600 | no |
| `unlockedacademy.co` | TXT | `unlockedacademy.co` | 3600 | no |
| `_dmarc.unlockedacademy.co` | TXT | `v=DMARC1; p=none` | 3600 | no |
| `www.unlockedacademy.co` | CNAME | `cname.learnworlds.com` | 3600 | no |
| `n8n.unlockedacademy.co` | CNAME | `b1ea5480-4905-4522-9c45-529a64b56e66.cfargotunnel.com` | auto | sí |

Cloudflare Free rechazó los TTL heredados de 30 segundos para los dos A del
apex; la API impuso su mínimo de 60 segundos. Los TTL restantes son los del
inventario. Un CNAME proxied usa obligatoriamente TTL `auto` (valor API `1`).

### Host y preflight

El volumen que contiene
`/mnt/data2tb/services/cloudflared/config.yml` estaba montado de solo lectura
para este ejecutor. La creación previa exigida de
`config.yml.bak-20260919T125802Z` falló con `Read-only file system`, antes de
cualquier cambio al archivo. Por ello **no existe backup nuevo**, no se añadió
el ingress de n8n y no se reinició `cloudflared-agentes`.

- La configuración actual sí valida con
  `cloudflared tunnel --config /mnt/data2tb/services/cloudflared/config.yml ingress validate`.
- La unidad continúa `active (running)` sin reinicio.
- N8n local respondió `200` en `http://127.0.0.1:5678/healthz`.
- Los hostnames existentes continuaron respondiendo: `school` 200, `crm` 200,
  `control` 307.
- El preflight contra un edge Cloudflare con `--resolve` no pudo completar TLS
  (`curl` 35, alert de handshake). La zona aún es `pending` y no tiene
  Universal SSL comprobable con este token, por lo que debe considerarse
  Universal SSL pendiente hasta que se deleguen los nameservers y se emita el
  certificado.

`ops/n8n/docker-compose.yml` ya contenía los valores requeridos de
`N8N_HOST`, `WEBHOOK_URL` y `N8N_EDITOR_BASE_URL`, todos apuntando a
`https://n8n.unlockedacademy.co`; no necesitó modificación.
