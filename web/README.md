# Web de resultados (maqueta local)

`build_web.py` lee el grafo (solo lectura) y genera una web estática en `dist/`:
portada con (1) tiempo de política por cadena, (2) reparto del tiempo de mención por partido y
(3) NPS de cada partido cadena a cadena con desplegable; página por partido,
página de evidencia por cadena × partido × mes, metodología, `datos/{politica,exposicion,nps}.csv|json`,
`llms.txt`, `sitemap.xml` y `robots.txt`.

## Verla en el NAS

Copiar `build_web.py` a `/volume2/docker/Proyecto_TV/web/` y, desde `/volume2/docker/Proyecto_TV`:

```bash
# 1) Solo el diseño, con datos ficticios (no toca el grafo; no necesita el driver)
docker run --rm -v "$PWD/web:/web" -p 8088:8088 python:3.12-slim \
  python -u /web/build_web.py --demo --out /web/dist --serve 8088

# 2) Con los datos reales: usa la imagen y el .env del capataz, dentro de la red del compose
docker compose run --rm --no-deps -v "$PWD/web:/web" -p 8088:8088 tone_analyzer \
  python -u /web/build_web.py --out /web/dist --serve 8088
```

Abrir `http://192.168.68.62:8088` (LAN) o `http://100.99.216.57:8088` (Tailscale). Ctrl+C para parar.

Conexión: `NEO4J_URI` (por defecto `bolt://neo4j:7687`), `NEO4J_USER` (`neo4j`), `NEO4J_PASS`.

Al arrancar con datos reales imprime la tabla `tipo_atribucion -> vía`: comprobar que cada valor
cae en la vía correcta (explícita, institucional, caso, alusiva) y, si no, ajustar `via_de()`.

## Pendiente antes de publicar

- Filtro del 90 % de programas completos por mes (§9.2): aún no aplicado.
- El texto de evidencia sale tal cual del grafo (`MENCION_PIEZA.texto`): revisar extensión y
  datos de particulares antes de publicarlo.
- `SITE_URL` (variable de entorno) con el dominio definitivo para `sitemap.xml` y `llms.txt`.
