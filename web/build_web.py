#!/usr/bin/env python3
"""Genera la web estática de Proyecto_TV a partir del grafo (o de datos ficticios).

Uso en el NAS (dentro de la red de docker, donde el grafo se llama `neo4j`):

    python build_web.py --out dist                 # lee Neo4j (NEO4J_URI/NEO4J_USER/NEO4J_PASS)
    python build_web.py --out dist --serve 8088    # y la sirve en http://<nas>:8088

Para probar sin grafo:

    python build_web.py --demo --out dist --serve 8088

Solo lee: no escribe nada en el grafo. Sin dependencias salvo el driver `neo4j`
(que ya lleva la imagen del capataz).
"""
import argparse
import collections
import csv
import datetime as dt
import html
import http.server
import json
import os
import random
import re
import shutil
import socketserver
import unicodedata

# ---------------------------------------------------------------- configuración

DESDE = "2026-02-01"  # ventana comparable (A.8): Antena3 y laSexta existen desde febrero
MIN_PAREJAS = 20      # §9.2: por debajo, la celda va vacía

PARTIDOS = ["PSOE", "PP", "Vox", "Sumar", "Podemos", "Izquierda Unida", "ERC", "Junts",
            "EH Bildu", "PNV", "Coalición Canaria", "Compromís", "BNG",
            "Chunta Aragonesista", "Ciudadanos", "Se Acabó La Fiesta"]
NACIONALES = ["RTVE", "Antena3", "laSexta"]           # entran en el agregado nacional
TERTULIAS = ["Malas Lenguas", "Todo es mentira"]      # A.8: nunca mezclar con informativos
AUTONOMICAS = ["Telemadrid"]                          # §9.2: fuera del agregado
CADENAS = NACIONALES + AUTONOMICAS + TERTULIAS

LECTURAS = {
    "todas": "Cuatro vías (indicador publicado)",
    "explicita": "Solo menciones explícitas",
}
VIAS = ["explícita", "institucional", "caso", "alusiva"]

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

SITE_URL = os.environ.get("SITE_URL", "https://example.org")  # cambiar al dominio real


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def mes_largo(m):
    y, mm = m.split("-")
    return f"{MESES[int(mm) - 1]} de {y}"


def via_de(tipo_atribucion):
    """Traduce tipo_atribucion del grafo a una de las cuatro vías (§9.3)."""
    t = (tipo_atribucion or "").lower()
    if "caso" in t:
        return "caso"
    if "alus" in t:
        return "alusiva"
    if any(k in t for k in ("instit", "gobierno", "control", "cargo")):
        return "institucional"
    return "explícita"


# ---------------------------------------------------------------- carga de datos

CYPHER = """
MATCH (b:Boletin)-[:TIENE_PIEZA]->(p:Pieza)-[m:MENCION_PIEZA]->(pt:Partido)
WHERE toString(b.fecha) >= $desde
OPTIONAL MATCH (p)-[v:VALORACION]->(pt)
RETURN toString(b.fecha) AS fecha, b.cadena AS cadena, b.archivo AS archivo,
       p.pieza_id AS pieza_id, p.tema AS tema, p.tipo AS tipo, p.t_ini AS p_ini, p.t_fin AS p_fin,
       coalesce(pt.federacion, pt.nombre) AS partido,
       m.tipo_atribucion AS tipo_atribucion, m.entidad AS entidad, m.texto AS texto,
       m.dur_evidencia AS dur, m.t_ini AS t_ini, m.t_fin AS t_fin, m.votos AS votos,
       v.tono_gemini AS tono_gemini, v.tono_qwen AS tono_qwen, v.tono_arbitro AS tono_arbitro,
       v.tono_final AS tono_final, v.rol AS rol, v.origen_tono AS origen_tono,
       v.justificacion AS justificacion, v.acuerdo AS acuerdo, v.version AS version
"""


def cargar_neo4j():
    from neo4j import GraphDatabase
    uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASS") or os.environ.get("NEO4J_PASSWORD")
    with GraphDatabase.driver(uri, auth=(user, pwd)) as drv:
        with drv.session(default_access_mode="READ") as s:
            rows = [dict(r) for r in s.run(CYPHER, desde=DESDE)]
    cnt = collections.Counter(r["tipo_atribucion"] for r in rows)
    print("tipo_atribucion -> vía (revisar que el mapeo es correcto):")
    for k, n in cnt.most_common():
        print(f"  {n:7d}  {k!r:40} -> {via_de(k)}")
    return rows


def cargar_demo():
    """Datos FICTICIOS, solo para ver la maqueta. No significan nada."""
    rnd = random.Random(7)
    temas = ["Presupuestos", "Vivienda", "Incendios forestales", "Financiación autonómica",
             "Inmigración", "Sanidad pública", "Caso judicial", "Pleno del Congreso",
             "Paro registrado", "Transporte ferroviario"]
    rows = []
    hoy = dt.date(2026, 9, 17)
    d = dt.date.fromisoformat(DESDE)
    while d <= hoy:
        for cadena in CADENAS:
            if cadena == "Todo es mentira":
                continue
            for k in range(rnd.randint(4, 9)):
                archivo = f"{d}_{slug(cadena)}_demo.json"
                ini = k * 40
                tema = rnd.choice(temas)
                for partido in rnd.sample(PARTIDOS[:6], rnd.randint(1, 3)) + \
                        ([rnd.choice(PARTIDOS[6:])] if rnd.random() < .15 else []):
                    tipo_at = rnd.choices(["censo_nombre", "institucion_mapa", "caso", "alusion"],
                                          [60, 30, 5, 5])[0]
                    tonos = ["negativo", "neutral", "positivo", "mixto"]
                    tg = rnd.choices(tonos, [38, 45, 12, 5])[0]
                    tq = tg if rnd.random() < .72 else rnd.choice(tonos)
                    ta = rnd.choice([tg, tq]) if tg != tq else None
                    rows.append(dict(
                        fecha=str(d), cadena=cadena, archivo=archivo,
                        pieza_id=f"{archivo}#{ini}-{ini + 30}", tema=tema, tipo="politica",
                        p_ini=ini * 4.0, p_fin=ini * 4.0 + 120,
                        partido=partido, tipo_atribucion=tipo_at, entidad="(ficticio)",
                        texto=">> Texto de ejemplo de la evidencia: aquí irían los segmentos de la "
                              "transcripción que sustentan la mención.\nSegmento de contexto sin marcar.",
                        dur=round(rnd.uniform(3, 60), 1), t_ini=ini * 4.0, t_fin=ini * 4.0 + 30,
                        votos=rnd.choice(["2/2", "3/3", "2/3"]),
                        tono_gemini=tg, tono_qwen=tq, tono_arbitro=ta, tono_final=ta or tg,
                        rol=rnd.choice(["atacante", "atacado", "neutro"]), origen_tono="directo",
                        justificacion="Justificación de ejemplo del evaluador.",
                        acuerdo=tg == tq, version="v4"))
        d += dt.timedelta(days=rnd.choice([1, 2]))
    return rows


def normalizar(rows):
    out = []
    for r in rows:
        if r["partido"] not in PARTIDOS or r["cadena"] not in CADENAS:
            continue
        r = dict(r)
        r["fecha"] = (r["fecha"] or "")[:10]
        r["mes"] = r["fecha"][:7]
        r["via"] = via_de(r["tipo_atribucion"])
        r["dur"] = float(r["dur"] or 0)
        out.append(r)
    return out


# ---------------------------------------------------------------- agregación

def ambitos(cadena):
    yield cadena
    if cadena in NACIONALES:
        yield "nacional"


def agregar(rows):
    """{(ambito, partido, mes, lectura): {n, seg, pos, neg}} solo con tono_final."""
    acc = collections.defaultdict(lambda: dict(n=0, seg=0.0, pos=0.0, neg=0.0, npos=0, nneg=0))
    for r in rows:
        tono = r["tono_final"]
        if not tono:
            continue
        lecturas = ["todas"] + (["explicita"] if r["via"] == "explícita" else [])
        for amb in ambitos(r["cadena"]):
            for mes in (r["mes"], "total"):
                for lec in lecturas:
                    a = acc[(amb, r["partido"], mes, lec)]
                    a["n"] += 1
                    a["seg"] += r["dur"]
                    if tono == "positivo":
                        a["pos"] += r["dur"]; a["npos"] += 1
                    elif tono == "negativo":
                        a["neg"] += r["dur"]; a["nneg"] += 1
    return acc


def nps(a):
    if not a or a["n"] < MIN_PAREJAS or a["seg"] <= 0:
        return None
    return 100 * (a["pos"] - a["neg"]) / a["seg"]


# ---------------------------------------------------------------- HTML

CSS = """
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#7a7974;--line:#e6e5e0;
--pos:#2a78d6;--neg:#e34948;--mid:#f0efec;--accent:#2a78d6;--warn-bg:#fff4d6;--warn-ink:#6b4e00;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--line:#2e2e2b;--pos:#3987e5;--neg:#e66767;--mid:#383835;
--accent:#6da7ec;--warn-bg:#3a2f0f;--warn-ink:#f3d58a}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;
--line:#2e2e2b;--pos:#3987e5;--neg:#e66767;--mid:#383835;--accent:#6da7ec;--warn-bg:#3a2f0f;--warn-ink:#f3d58a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
header.top{border-bottom:1px solid var(--line);background:var(--surface)}
header.top .wrap{display:flex;gap:24px;align-items:center;justify-content:space-between;padding-top:14px;padding-bottom:14px;flex-wrap:wrap}
.brand{font-weight:700;letter-spacing:-.01em;text-decoration:none;color:var(--ink);font-size:18px}
nav a{color:var(--ink2);text-decoration:none;margin-left:18px;font-size:15px}
nav a:hover{color:var(--ink)}
.demo{background:var(--warn-bg);color:var(--warn-ink);text-align:center;padding:8px 16px;font-weight:600;font-size:14px}
h1{font-size:clamp(26px,4vw,38px);line-height:1.15;letter-spacing:-.02em;margin:36px 0 10px}
h2{font-size:22px;margin:44px 0 6px;letter-spacing:-.01em}
h3{font-size:17px;margin:24px 0 6px}
.lede{font-size:18px;color:var(--ink2);max-width:760px}
.muted{color:var(--ink3);font-size:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:16px 0}
.card>:first-child{margin-top:0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .l{color:var(--ink2);font-size:14px}
.tbl{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl th{color:var(--ink2);font-weight:600;font-size:13px}
.scroll{overflow-x:auto}
.cell{display:inline-block;min-width:46px;padding:2px 6px;border-radius:4px}
.empty{color:var(--ink3)}
svg text{fill:var(--ink2);font:12px system-ui,sans-serif}
svg .lbl{fill:var(--ink)}
.bar-pos{fill:var(--pos)}.bar-neg{fill:var(--neg)}.axis{stroke:var(--ink3);stroke-width:1}
.grid{stroke:var(--line);stroke-width:1}
.bar:hover{opacity:.8}
.ev{border-top:1px solid var(--line);padding:14px 0}
.ev:first-child{border-top:0}
.ev .meta{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--ink2)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:0 8px;font-size:12px;color:var(--ink2)}
.tono{font-weight:600}.tono.negativo{color:var(--neg)}.tono.positivo{color:var(--pos)}
.txt{font-family:var(--mono);font-size:13px;white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;margin-top:8px}
.txt mark{background:transparent;color:var(--ink);font-weight:700}
.votes{font-size:13px;color:var(--ink2);margin-top:6px}
footer{border-top:1px solid var(--line);margin-top:60px;padding:24px 0;color:var(--ink3);font-size:14px}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:6px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.pill{display:inline-block;padding:3px 10px;border:1px solid var(--line);border-radius:999px;margin:3px 4px 3px 0;text-decoration:none;color:var(--ink2);font-size:14px;background:var(--surface)}
.pill:hover{border-color:var(--accent);color:var(--ink)}
"""

E = html.escape


def fmt(x, signo=True):
    if x is None:
        return '<span class="empty">—</span>'
    return f"{x:+.1f}".replace(".", ",") if signo else f"{x:.1f}".replace(".", ",")


def miles(n):
    return f"{n:,.0f}".replace(",", ".")


def celda(x):
    """Celda con fondo divergente suave (el número siempre visible)."""
    if x is None:
        return '<span class="empty">—</span>'
    k = min(abs(x) / 40, 1) * 0.28
    col = "var(--pos)" if x >= 0 else "var(--neg)"
    return (f'<span class="cell" style="background:color-mix(in srgb,{col} {k * 100:.0f}%,transparent)">'
            f'{fmt(x)}</span>')


class Sitio:
    def __init__(self, out, demo, rows, acc):
        self.out, self.demo, self.rows, self.acc = out, demo, rows, acc
        self.meses = sorted({r["mes"] for r in rows})
        self.urls = []
        self.generado = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # -- utilidades
    def write(self, rel, content):
        path = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if rel.endswith(".html"):
            self.urls.append(rel.replace("index.html", ""))

    def page(self, rel, title, desc, body, jsonld=None):
        depth = rel.count("/")
        root = "../" * depth
        demo = ('<div class="demo">DATOS FICTICIOS — maqueta para ver el diseño; las cifras no '
                'significan nada</div>') if self.demo else ""
        ld = f'<script type="application/ld+json">{json.dumps(jsonld, ensure_ascii=False)}</script>' if jsonld else ""
        self.write(rel, f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{E(title)}</title><meta name="description" content="{E(desc)}">
<link rel="alternate" type="text/plain" href="{root}llms.txt" title="llms.txt">
<style>{CSS}</style>{ld}</head><body>{demo}
<header class="top"><div class="wrap"><a class="brand" href="{root}index.html">Observatorio de telediarios</a>
<nav><a href="{root}index.html">Resultados</a><a href="{root}partidos/index.html">Partidos</a>
<a href="{root}metodologia/index.html">Metodología</a><a href="{root}datos/index.html">Datos abiertos</a></nav></div></header>
<main class="wrap">{body}</main>
<footer><div class="wrap">Generado el {self.generado}. Cifras: NPS ponderado por tiempo de evidencia,
de −100 (todo el tiempo de mención es negativo) a +100. Celdas con menos de {MIN_PAREJAS} parejas, vacías.
<a href="{root}metodologia/index.html">Cómo se mide</a> · <a href="{root}datos/index.html">Descargar los datos</a></div></footer>
</body></html>""")

    def v(self, amb, partido, mes="total", lec="todas"):
        return nps(self.acc.get((amb, partido, mes, lec)))

    def n(self, amb, partido, mes="total", lec="todas"):
        a = self.acc.get((amb, partido, mes, lec))
        return a["n"] if a else 0

    # -- gráficos
    def diverging(self, items, titulo):
        """items: [(etiqueta, valor, n)] -> barras horizontales divergentes en SVG."""
        items = [i for i in items if i[1] is not None]
        if not items:
            return '<p class="muted">Sin datos suficientes.</p>'
        m = max(10, max(abs(v) for _, v, _ in items))
        m = (int(m / 10) + 1) * 10
        W, L, R, rowh = 520, 150, 44, 26
        plot = W - L - R
        H = rowh * len(items) + 30
        x0 = L + plot / 2
        sx = lambda v: x0 + v / m * plot / 2
        g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{E(titulo)}">']
        for t in (-m, -m / 2, 0, m / 2, m):
            x = sx(t)
            g.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{H - 22}"/>')
            g.append(f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle">{t:+.0f}</text>')
        for i, (lab, v, n) in enumerate(items):
            y = i * rowh + 4
            x1, x2 = sorted((x0, sx(v)))
            w = max(x2 - x1, 1)
            cls = "bar-pos" if v >= 0 else "bar-neg"
            g.append(f'<g class="bar"><title>{E(lab)}: {fmt(v)} ({n} parejas)</title>'
                     f'<rect class="{cls}" x="{x1:.1f}" y="{y + 4}" width="{w:.1f}" height="{rowh - 10}" rx="4"/></g>')
            g.append(f'<text class="lbl" x="{L - 8}" y="{y + rowh / 2 + 3}" text-anchor="end">{E(lab)}</text>')
            tx = sx(v) + (6 if v >= 0 else -6)
            g.append(f'<text x="{tx:.1f}" y="{y + rowh / 2 + 3}" text-anchor="{"start" if v >= 0 else "end"}">{fmt(v)}</text>')
        g.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="0" y2="{H - 22}"/></svg>')
        return "".join(g)

    def serie_mensual(self, amb, partido, lec="todas", mx=None):
        """Barras verticales divergentes por mes (mx: escala común para comparar cadenas)."""
        vals = [(m, self.v(amb, partido, m, lec)) for m in self.meses]
        if all(v is None for _, v in vals):
            return '<p class="muted">Sin meses con datos suficientes.</p>'
        mx = mx or max(10, max(abs(v) for _, v in vals if v is not None))
        mx = (int(mx / 10) + 1) * 10
        W, H, T, B, L = 1000, 190, 10, 26, 40
        ph = H - T - B
        bw = (W - L) / len(vals)
        y0 = T + ph / 2
        sy = lambda v: y0 - v / mx * ph / 2
        g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="NPS mensual {E(partido)} en {E(amb)}">']
        for t in (-mx, 0, mx):
            g.append(f'<line class="grid" x1="{L}" x2="{W}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                     f'<text x="{L - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:+.0f}</text>')
        for i, (m, v) in enumerate(vals):
            x = L + i * bw
            g.append(f'<text x="{x + bw / 2:.1f}" y="{H - 10}" text-anchor="middle">{MESES[int(m[5:]) - 1][:3]}</text>')
            if v is None:
                continue
            y1, y2 = sorted((y0, sy(v)))
            cls = "bar-pos" if v >= 0 else "bar-neg"
            g.append(f'<g class="bar"><title>{mes_largo(m)}: {fmt(v)} ({self.n(amb, partido, m, lec)} parejas)</title>'
                     f'<rect class="{cls}" x="{x + bw * .2:.1f}" y="{y1:.1f}" width="{bw * .6:.1f}" height="{max(y2 - y1, 1):.1f}" rx="3"/></g>')
        g.append(f'<line class="axis" x1="{L}" x2="{W}" y1="{y0}" y2="{y0}"/></svg>')
        return "".join(g)

    # -- páginas
    def index(self):
        periodo = f"{mes_largo(self.meses[0])} a {mes_largo(self.meses[-1])}" if self.meses else ""
        pares = [(p, self.v("nacional", p), self.n("nacional", p)) for p in PARTIDOS]
        pares.sort(key=lambda t: (t[1] is None, -(t[2])))
        # los 8 con más parejas, mostrados siempre en el orden fijo de PARTIDOS
        top = sorted([t for t in pares if t[1] is not None][:8], key=lambda t: PARTIDOS.index(t[0]))
        orden = [t[0] for t in top]
        exp = [(p, self.v("nacional", p, lec="explicita"), self.n("nacional", p, lec="explicita")) for p in orden]
        dif = lambda amb, lec="todas": (None if None in (self.v(amb, "PSOE", lec=lec), self.v(amb, "PP", lec=lec))
                                        else self.v(amb, "PSOE", lec=lec) - self.v(amb, "PP", lec=lec))
        n_parejas = sum(1 for r in self.rows if r["tono_final"])
        n_piezas = len({r["pieza_id"] for r in self.rows})
        seg = sum(r["dur"] for r in self.rows if r["tono_final"])

        filas = []
        for c in NACIONALES + AUTONOMICAS + TERTULIAS:
            nota = " <span class='tag'>tertulia</span>" if c in TERTULIAS else (
                " <span class='tag'>autonómica</span>" if c in AUTONOMICAS else "")
            celdas = "".join(f"<td>{celda(self.v(c, p))}</td>" for p in orden)
            filas.append(f"<tr><td>{E(c)}{nota}</td>{celdas}<td>{fmt(dif(c))}</td></tr>")
        cab = "".join(f'<th><a href="partidos/{slug(p)}/index.html">{E(p)}</a></th>' for p in orden)

        body = f"""
<h1>Cómo tratan los telediarios a cada partido</h1>
<p class="lede">Medimos, noticia a noticia, qué imagen transmite cada telediario sobre cada partido.
Dos modelos de IA valoran cada caso por separado, un tercero arbitra los desacuerdos sin saberlo,
y <strong>cada cifra enlaza con los fragmentos que la sustentan</strong>. Periodo: {periodo}.</p>
<div class="kpis">
 <div class="kpi"><div class="v">{miles(n_piezas)}</div><div class="l">noticias analizadas</div></div>
 <div class="kpi"><div class="v">{miles(n_parejas)}</div><div class="l">parejas noticia–partido valoradas</div></div>
 <div class="kpi"><div class="v">{miles(seg / 3600)} h</div><div class="l">de evidencia citada</div></div>
 <div class="kpi"><div class="v">{fmt(dif("nacional"))}</div><div class="l">diferencial PSOE − PP (nacional)</div></div>
</div>
<h2>NPS por partido · cadenas nacionales</h2>
<p class="muted">RTVE, Antena3 y laSexta. Sin tertulias ni autonómicas. Pasa el ratón por una barra para ver cuántas parejas la sustentan.</p>
<div class="legend"><span><i style="background:var(--pos)"></i>trato positivo neto</span><span><i style="background:var(--neg)"></i>trato negativo neto</span></div>
<div class="grid2">
 <div class="card"><h3>{LECTURAS["todas"]}</h3>{self.diverging(top, LECTURAS["todas"])}</div>
 <div class="card"><h3>{LECTURAS["explicita"]}</h3>{self.diverging(exp, LECTURAS["explicita"])}</div>
</div>
<p class="muted">Por qué mostramos las dos: quien gobierna aparece mucho más por vía institucional
("el Gobierno", "la Junta"). Con las cuatro vías el diferencial PSOE − PP es {fmt(dif("nacional"))};
solo con menciones explícitas, {fmt(dif("nacional", "explicita"))}. <a href="metodologia/index.html#vias">Más detalle</a>.</p>

<h2>Cadena a cadena</h2>
<p class="muted">Compara siempre dentro de la misma cadena. Las tertulias tienen otro formato y no se suman a los informativos.</p>
<div class="card scroll"><table class="tbl"><thead><tr><th>Cadena</th>{cab}<th>PSOE − PP</th></tr></thead>
<tbody>{"".join(filas)}</tbody></table></div>
"""
        self.page("index.html", "Cómo tratan los telediarios a cada partido",
                  "NPS ponderado por tiempo del trato de RTVE, Antena3, laSexta, Telemadrid y tertulias a cada partido, con la evidencia de cada valoración.",
                  body, {"@context": "https://schema.org", "@type": "Dataset",
                         "name": "Trato de los telediarios españoles a los partidos políticos",
                         "description": "NPS ponderado por tiempo, por cadena, partido y mes, con evidencia por noticia.",
                         "temporalCoverage": f"{self.meses[0]}/{self.meses[-1]}" if self.meses else None,
                         "license": "https://creativecommons.org/licenses/by/4.0/",
                         "distribution": [{"@type": "DataDownload", "encodingFormat": "text/csv",
                                           "contentUrl": f"{SITE_URL}/datos/nps.csv"}]})

    def partidos(self):
        pills = "".join(f'<a class="pill" href="{slug(p)}/index.html">{E(p)}</a>' for p in PARTIDOS)
        self.page("partidos/index.html", "Partidos", "Resultados por partido.",
                  f"<h1>Partidos</h1><p class='lede'>Elige un partido para ver su serie mensual cadena a cadena.</p><div>{pills}</div>")
        for p in PARTIDOS:
            bloques = []
            todos = [self.v(c, p, m) for c in ["nacional"] + CADENAS for m in self.meses]
            mx = max([10] + [abs(x) for x in todos if x is not None])
            for c in ["nacional"] + CADENAS:
                if self.n(c, p) == 0:
                    continue
                meses = " ".join(
                    f'<a class="pill" href="../../evidencia/{slug(c)}/{slug(p)}/{m}.html">{MESES[int(m[5:]) - 1][:3]} {fmt(self.v(c, p, m))}</a>'
                    for m in self.meses if c != "nacional" and self.n(c, p, m))
                nombre = "Agregado nacional (RTVE, Antena3, laSexta)" if c == "nacional" else c
                bloques.append(f"""<div class="card"><h3>{E(nombre)} · {fmt(self.v(c, p))}
<span class="muted">({miles(self.n(c, p))} parejas; solo explícitas: {fmt(self.v(c, p, lec="explicita"))})</span></h3>
{self.serie_mensual(c, p, mx=mx)}{"<p class='muted'>Evidencia por mes:</p>" + meses if meses else ""}</div>""")
            vtot = self.v("nacional", p)
            frase = (f"En las cadenas nacionales, entre {mes_largo(self.meses[0])} y {mes_largo(self.meses[-1])}, "
                     f"{p} recibió un NPS ponderado de {fmt(vtot)} sobre {miles(self.n('nacional', p))} parejas noticia–partido."
                     if vtot is not None else f"No hay parejas suficientes de {p} en el agregado nacional.")
            self.page(f"partidos/{slug(p)}/index.html", f"{p}: trato en los telediarios",
                      frase, f"<h1>{E(p)}</h1><p class='lede'>{E(frase)}</p>{''.join(bloques)}")

    def evidencia(self):
        grupos = collections.defaultdict(list)
        for r in self.rows:
            grupos[(r["cadena"], r["partido"], r["mes"])].append(r)
        for (c, p, m), rs in grupos.items():
            rs.sort(key=lambda r: (r["fecha"], r["pieza_id"] or ""))
            v, n = self.v(c, p, m), self.n(c, p, m)
            frase = (f"En {mes_largo(m)}, {c} dio a {p} un NPS ponderado de {fmt(v)} "
                     f"sobre {n} parejas noticia–partido valoradas." if v is not None else
                     f"En {mes_largo(m)}, {c} tiene {n} parejas valoradas de {p}: por debajo del mínimo de {MIN_PAREJAS} para publicar cifra.")
            por_via = collections.Counter(r["via"] for r in rs)
            items = []
            for r in rs:
                texto = E(r["texto"] or "")
                texto = re.sub(r"(?m)^&gt;&gt;(.*)$", r"<mark>▸\1</mark>", texto)
                arb = f" · árbitro: <span class='tono {E(r['tono_arbitro'] or '')}'>{E(r['tono_arbitro'])}</span>" if r["tono_arbitro"] else ""
                items.append(f"""<div class="ev" id="{E(slug(r['pieza_id'] or ''))}">
<div><strong>{E(r['fecha'])}</strong> · {E(r['tema'] or 'sin tema')} ·
<span class="tono {E(r['tono_final'] or '')}">{E(r['tono_final'] or 'pendiente')}</span></div>
<div class="meta"><span class="tag">vía {E(r['via'])}</span><span>entidad: {E(str(r['entidad'] or ''))}</span>
<span>rol: {E(r['rol'] or '—')}</span><span>evidencia: {r['dur']:.0f} s</span><span>detección: {E(str(r['votos'] or ''))}</span></div>
<div class="votes">Gemini: {E(r['tono_gemini'] or '—')} · Qwen: {E(r['tono_qwen'] or '—')}{arb}
{' · ' + E(r['justificacion']) if r['justificacion'] else ''}</div>
<div class="txt">{texto}</div>
<div class="muted">Pieza <code>{E(r['pieza_id'] or '')}</code> · versión {E(str(r['version'] or ''))}</div></div>""")
            vias = " · ".join(f"{k}: {por_via[k]}" for k in VIAS if por_via[k])
            self.page(f"evidencia/{slug(c)}/{slug(p)}/{m}.html", f"{p} en {c}, {mes_largo(m)}", frase,
                      f"""<p class="muted"><a href="../../../partidos/{slug(p)}/index.html">← {E(p)}</a></p>
<h1>{E(p)} en {E(c)} · {mes_largo(m)}</h1><p class="lede">{E(frase)}</p>
<div class="kpis"><div class="kpi"><div class="v">{fmt(v)}</div><div class="l">NPS ponderado (cuatro vías)</div></div>
<div class="kpi"><div class="v">{fmt(self.v(c, p, m, "explicita"))}</div><div class="l">solo explícitas</div></div>
<div class="kpi"><div class="v">{len(rs)}</div><div class="l">menciones ({vias})</div></div></div>
<h2>La evidencia, una a una</h2><p class="muted">Las líneas marcadas con ▸ son los segmentos que sustentan la mención y dan su peso.</p>
<div class="card">{''.join(items)}</div>""")

    def metodologia(self):
        body = """<h1>Cómo se mide</h1>
<p class="lede">La unidad es la pareja (noticia, partido). Para cada una se decide si la noticia habla del partido
y qué imagen transmite sobre él. El indicador es el NPS ponderado por tiempo.</p>
<div class="card"><h3>La fórmula</h3><p><code>NPS = 100 · (segundos positivos − segundos negativos) / segundos totales</code></p>
<p>Los segundos son los de los fragmentos que sustentan cada mención, no los de la noticia entera: una mención de paso pesa poco.
Las valoraciones neutrales y mixtas cuentan en el denominador y empujan la cifra hacia cero.</p></div>
<div class="card"><h3>Qué es positivo y qué negativo</h3><ul>
<li><strong>Negativo</strong>: escándalos, casos judiciales, errores de gestión o malos resultados donde el partido gobierna; o que otro actor le critique o acuse.</li>
<li><strong>Positivo</strong>: solo con un resultado afirmado (un indicador que mejora, un problema resuelto) o un elogio explícito.</li>
<li><strong>Neutral</strong>: anunciar, reunirse, presentar un plan. La gestión rutinaria no puntúa.</li>
<li><strong>El que ataca queda neutral; el atacado, negativo.</strong></li>
<li>Los gobiernos autonómicos y municipales cuentan igual que el central.</li></ul></div>
<div class="card"><h3>Quién decide</h3><p>La IA nunca decide a qué partido pertenece una persona o institución:
devuelve la entidad tal como aparece ("el Gobierno", "Ayuso") y es código determinista el que la cruza con tablas fechadas
(quién gobernaba cada territorio ese día). El tono lo valoran dos modelos distintos por separado; si discrepan, un tercero
lo valora sin saber que arbitra y el resultado sale por mayoría.</p></div>
<div class="card" id="vias"><h3>Las cuatro vías de mención</h3><p>Un partido puede aparecer porque se le nombra
(<em>explícita</em>), porque se habla de una institución que gobierna (<em>institucional</em>), por un caso judicial que le
implica (<em>caso</em>) o por una alusión ("Ferraz", "Génova"). Quien gobierna el Estado acumula muchas más menciones
institucionales, y eso puede cambiar el signo de las comparaciones. Por eso publicamos las dos lecturas.</p></div>
<div class="card"><h3>Filtros</h3><ul><li>Desde febrero de 2026, primer mes con las tres cadenas nacionales.</li>
<li>Menos de 20 parejas en una celda: no se publica cifra.</li>
<li>Telemadrid se muestra aparte (autonómica). Las tertulias no se suman a los informativos.</li></ul></div>
<div class="card"><h3>Limitaciones conocidas</h3><ul><li>No hay Telecinco ni Cuatro (contenido protegido).</li>
<li>La transcripción automática comete errores en nombres propios.</li>
<li>El sistema va con hasta dos días de retraso sobre la emisión.</li></ul></div>"""
        self.page("metodologia/index.html", "Metodología", "Cómo se mide el trato de los telediarios a los partidos.", body)

    def datos(self):
        filas = []
        for (amb, p, m, lec), a in sorted(self.acc.items()):
            filas.append(dict(ambito=amb, partido=p, mes=m, lectura=lec, parejas=a["n"],
                              seg_total=round(a["seg"], 1), seg_pos=round(a["pos"], 1), seg_neg=round(a["neg"], 1),
                              nps_ponderado=None if nps(a) is None else round(nps(a), 2)))
        os.makedirs(os.path.join(self.out, "datos"), exist_ok=True)
        with open(os.path.join(self.out, "datos", "nps.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]) if filas else ["ambito"])
            w.writeheader(); w.writerows(filas)
        with open(os.path.join(self.out, "datos", "nps.json"), "w", encoding="utf-8") as f:
            json.dump({"generado": self.generado, "min_parejas": MIN_PAREJAS, "filas": filas}, f, ensure_ascii=False)
        self.page("datos/index.html", "Datos abiertos", "Descarga de los indicadores.",
                  f"""<h1>Datos abiertos</h1><p class="lede">Los indicadores, con licencia CC BY 4.0.</p>
<div class="card"><ul><li><a href="nps.csv">nps.csv</a> — ámbito, partido, mes, lectura, parejas, segundos y NPS.</li>
<li><a href="nps.json">nps.json</a> — lo mismo en JSON.</li></ul>
<p class="muted">{miles(len(filas))} filas. <code>mes = total</code> es el acumulado del periodo.</p></div>""")

    def llms(self):
        dif = None
        if None not in (self.v("nacional", "PSOE"), self.v("nacional", "PP")):
            dif = self.v("nacional", "PSOE") - self.v("nacional", "PP")
        lineas = [f"- {p}: {fmt(self.v('nacional', p))} (solo explícitas {fmt(self.v('nacional', p, lec='explicita'))})"
                  for p in PARTIDOS if self.v("nacional", p) is not None]
        lineas = [re.sub(r"<[^>]+>", "", l) for l in lineas]
        txt = f"""# Observatorio de telediarios

> Mide cómo tratan los telediarios españoles (RTVE, Antena3, laSexta, Telemadrid y tertulias) a cada partido.
> Indicador: NPS ponderado por tiempo de evidencia, de -100 a +100. Cada cifra enlaza con los fragmentos que la sustentan.

Periodo: {self.meses[0] if self.meses else ''} a {self.meses[-1] if self.meses else ''}. Generado: {self.generado}.
{"AVISO: DATOS FICTICIOS DE MAQUETA." if self.demo else ""}

## Cifras principales (agregado nacional: RTVE, Antena3, laSexta)
{chr(10).join(lineas)}
- Diferencial PSOE - PP: {re.sub(r"<[^>]+>", "", fmt(dif))}

## Páginas
- [Resultados]({SITE_URL}/index.html)
- [Metodología]({SITE_URL}/metodologia/index.html)
- [Partidos]({SITE_URL}/partidos/index.html): /partidos/<partido>/
- Evidencia por cadena, partido y mes: /evidencia/<cadena>/<partido>/<AAAA-MM>.html
- [Datos CSV]({SITE_URL}/datos/nps.csv) · [JSON]({SITE_URL}/datos/nps.json)

## Cómo citar
Observatorio de telediarios, NPS ponderado por tiempo, {self.generado[:10]}, licencia CC BY 4.0.
"""
        self.write("llms.txt", txt)
        self.write("robots.txt", f"User-agent: *\nAllow: /\nSitemap: {SITE_URL}/sitemap.xml\n")
        urls = "".join(f"<url><loc>{SITE_URL}/{E(u)}</loc></url>" for u in self.urls)
        self.write("sitemap.xml", f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>')

    def build(self):
        if os.path.isdir(self.out):
            shutil.rmtree(self.out)
        os.makedirs(self.out)
        self.index(); self.partidos(); self.evidencia(); self.metodologia(); self.datos(); self.llms()
        print(f"{len(self.urls)} páginas en {self.out}/ ({len(self.rows)} menciones, meses {self.meses[0]}..{self.meses[-1]})")


def servir(out, port):
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=out, **k)
    with socketserver.ThreadingTCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Sirviendo {out}/ en http://0.0.0.0:{port}  (Ctrl+C para parar)")
        httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    ap.add_argument("--demo", action="store_true", help="datos ficticios, sin grafo")
    ap.add_argument("--serve", type=int, metavar="PUERTO")
    a = ap.parse_args()
    rows = normalizar(cargar_demo() if a.demo else cargar_neo4j())
    if not rows:
        raise SystemExit("No hay menciones desde " + DESDE)
    Sitio(a.out, a.demo, rows, agregar(rows)).build()
    if a.serve:
        servir(a.out, a.serve)


if __name__ == "__main__":
    main()
