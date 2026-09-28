#!/usr/bin/env python3
"""Genera la web estática de Proyecto_TV a partir del grafo (o de datos ficticios).

Uso en el NAS (dentro de la red de docker, donde el grafo se llama `neo4j`):

    python build_web.py --out dist                 # lee Neo4j (NEO4J_URI/NEO4J_USER/NEO4J_PASS)
    python build_web.py --out dist --serve 8088    # y la sirve en http://<nas>:8088

Para probar sin grafo:

    python build_web.py --demo --out dist --serve 8088

Solo lee: no escribe nada en el grafo. Sin dependencias salvo el driver `neo4j`
(que ya lleva la imagen del capataz).

Qué publica, por decisión metodológica:
  1. Cuánto se habla de política en los telediarios, en total y por cadena.
  2. Cuánto se habla de cada partido dentro de la política, en total y por cadena.
  3. El NPS de cada partido cadena a cadena. El NPS NO se compara entre partidos:
     lo comparable es cómo trata cada cadena a un mismo partido.
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
NACIONAL = "Cadenas nacionales"                       # etiqueta del agregado (solo volumen, nunca NPS)

VIAS = ["explícita", "institucional", "caso", "alusiva"]

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
         "septiembre", "octubre", "noviembre", "diciembre"]

# Colores categóricos para el reparto por partido: identidad, no marca del partido
# (paleta validada para daltonismo en orden fijo; el resto se pliega en "Otros").
CATEG = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)"]
N_REPARTO = 5
# Un color fijo por cadena en el gráfico de evolución (nunca cambia al filtrar).
COLOR_CADENA = {"RTVE": "var(--c1)", "Antena3": "var(--c2)", "laSexta": "var(--c3)",
                "Telemadrid": "var(--c4)", "Malas Lenguas": "var(--c5)", "Todo es mentira": "var(--c6)"}

SITE_URL = os.environ.get("SITE_URL", "https://example.org")  # cambiar al dominio real


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def mes_largo(m):
    y, mm = m.split("-")
    return f"{MESES[int(mm) - 1]} de {y}"


def mes_corto(m):
    return MESES[int(m[5:]) - 1][:3]


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

CYPHER_MENCIONES = """
MATCH (b:Boletin)-[:TIENE_PIEZA]->(p:Pieza)-[m:MENCION_PIEZA]->(pt:Partido)
WHERE toString(b.fecha) >= $desde
OPTIONAL MATCH (p)-[v:VALORACION]->(pt)
RETURN toString(b.fecha) AS fecha, b.cadena AS cadena, b.archivo AS archivo,
       p.pieza_id AS pieza_id, p.tema AS tema, p.tipo AS tipo,
       coalesce(pt.federacion, pt.nombre) AS partido,
       m.tipo_atribucion AS tipo_atribucion, m.entidad AS entidad, m.texto AS texto,
       m.dur_evidencia AS dur, m.votos AS votos,
       v.tono_gemini AS tono_gemini, v.tono_qwen AS tono_qwen, v.tono_arbitro AS tono_arbitro,
       v.tono_final AS tono_final, v.rol AS rol, v.justificacion AS justificacion,
       v.acuerdo AS acuerdo, v.version AS version
"""

# Tiempo de política por boletín (como el bloque 8 del parte diario): piezas con al menos
# una mención de partido sobre la duración del boletín; solo boletines ya cortados.
CYPHER_BOLETINES = """
MATCH (b:Boletin)-[:TIENE_PIEZA]->(p:Pieza)
WHERE toString(b.fecha) >= $desde
OPTIONAL MATCH (p)-[m:MENCION_PIEZA]->(:Partido)
WITH b, p, count(m) > 0 AS politica
RETURN toString(b.fecha) AS fecha, b.cadena AS cadena, b.archivo AS archivo,
       b.duracion_seg AS duracion_seg, count(p) AS piezas, sum(p.dur) AS dur_piezas,
       sum(CASE WHEN politica THEN p.dur ELSE 0 END) AS dur_politica
"""


def cargar_neo4j():
    from neo4j import GraphDatabase
    uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASS") or os.environ.get("NEO4J_PASSWORD")
    with GraphDatabase.driver(uri, auth=(user, pwd)) as drv:
        with drv.session(default_access_mode="READ") as s:
            menciones = [dict(r) for r in s.run(CYPHER_MENCIONES, desde=DESDE)]
            boletines = [dict(r) for r in s.run(CYPHER_BOLETINES, desde=DESDE)]
    cnt = collections.Counter(r["tipo_atribucion"] for r in menciones)
    print("tipo_atribucion -> vía (revisar que el mapeo es correcto):")
    for k, n in cnt.most_common():
        print(f"  {n:7d}  {k!r:40} -> {via_de(k)}")
    sin_dur = sum(1 for b in boletines if not b["duracion_seg"])
    if sin_dur:
        print(f"AVISO: {sin_dur} boletines sin duracion_seg; se usa la suma de sus piezas")
    return menciones, boletines


def cargar_demo():
    """Datos FICTICIOS, solo para ver la maqueta. No significan nada."""
    rnd = random.Random(7)
    temas = ["Presupuestos", "Vivienda", "Incendios forestales", "Financiación autonómica",
             "Inmigración", "Sanidad pública", "Caso judicial", "Pleno del Congreso",
             "Paro registrado", "Transporte ferroviario"]
    peso_partido = [30, 28, 12, 8, 6] + [2] * 11
    pct_pol = {"RTVE": .28, "Antena3": .42, "laSexta": .36, "Telemadrid": .30, "Malas Lenguas": .70}
    menciones, boletines = [], []
    d = dt.date.fromisoformat(DESDE)
    while d <= dt.date(2026, 9, 17):
        for cadena in CADENAS:
            if cadena == "Todo es mentira":
                continue
            archivo = f"{d}_{slug(cadena)}_demo.json"
            dur_bol = rnd.uniform(3000, 3600) if cadena in TERTULIAS else rnd.uniform(1500, 2400)
            n_piezas = rnd.randint(12, 22)
            dur_pieza = dur_bol / n_piezas
            dur_pol = 0.0
            for k in range(n_piezas):
                if rnd.random() > pct_pol[cadena] * rnd.uniform(.8, 1.2):
                    continue
                dur_pol += dur_pieza
                ini = k * 40
                tema = rnd.choice(temas)
                elegidos = set(rnd.choices(PARTIDOS, peso_partido, k=rnd.randint(1, 3)))
                for partido in elegidos:
                    tonos = ["negativo", "neutral", "positivo", "mixto"]
                    tg = rnd.choices(tonos, [38, 45, 12, 5])[0]
                    tq = tg if rnd.random() < .72 else rnd.choice(tonos)
                    ta = rnd.choice([tg, tq]) if tg != tq else None
                    menciones.append(dict(
                        fecha=str(d), cadena=cadena, archivo=archivo,
                        pieza_id=f"{archivo}#{ini}-{ini + 30}", tema=tema, tipo="politica",
                        partido=partido,
                        tipo_atribucion=rnd.choices(["censo_nombre", "institucion_mapa", "caso", "alusion"],
                                                    [60, 30, 5, 5])[0],
                        entidad="(ficticio)",
                        texto=">> Texto de ejemplo de la evidencia: aquí irían los segmentos de la "
                              "transcripción que sustentan la mención.\nSegmento de contexto sin marcar.",
                        dur=round(rnd.uniform(3, min(60, dur_pieza)), 1),
                        votos=rnd.choice(["2/2", "3/3", "2/3"]),
                        tono_gemini=tg, tono_qwen=tq, tono_arbitro=ta, tono_final=ta or tg,
                        rol=rnd.choice(["atacante", "atacado", "neutro"]),
                        justificacion="Justificación de ejemplo del evaluador.",
                        acuerdo=tg == tq, version="v4"))
            boletines.append(dict(fecha=str(d), cadena=cadena, archivo=archivo, duracion_seg=dur_bol,
                                  piezas=n_piezas, dur_piezas=dur_bol, dur_politica=dur_pol))
        d += dt.timedelta(days=rnd.choice([1, 2]))
    return menciones, boletines


def normalizar(menciones, boletines):
    ms = []
    for r in menciones:
        if r["partido"] not in PARTIDOS or r["cadena"] not in CADENAS:
            continue
        r = dict(r)
        r["fecha"] = (r["fecha"] or "")[:10]
        r["mes"] = r["fecha"][:7]
        r["via"] = via_de(r["tipo_atribucion"])
        r["dur"] = float(r["dur"] or 0)
        ms.append(r)
    bs = []
    for b in boletines:
        if b["cadena"] not in CADENAS:
            continue
        b = dict(b)
        b["fecha"] = (b["fecha"] or "")[:10]
        b["mes"] = b["fecha"][:7]
        b["total"] = float(b["duracion_seg"] or b["dur_piezas"] or 0)
        b["politica"] = min(float(b["dur_politica"] or 0), b["total"])
        bs.append(b)
    return ms, bs


# ---------------------------------------------------------------- agregación

def ambitos(cadena):
    """Para volumen (política y reparto por partido): la cadena y, si es nacional, el agregado."""
    yield cadena
    if cadena in NACIONALES:
        yield NACIONAL


class Datos:
    def __init__(self, menciones, boletines):
        self.menciones, self.boletines = menciones, boletines
        self.meses = sorted({b["mes"] for b in boletines} | {r["mes"] for r in menciones})

        # 1. tiempo de política
        self.pol = collections.defaultdict(lambda: dict(total=0.0, politica=0.0, boletines=0))
        for b in boletines:
            for amb in ambitos(b["cadena"]):
                for mes in (b["mes"], "total"):
                    a = self.pol[(amb, mes)]
                    a["total"] += b["total"]; a["politica"] += b["politica"]; a["boletines"] += 1

        # 2. exposición: segundos de evidencia por partido (todas las menciones, valoradas o no)
        self.expo = collections.defaultdict(float)
        for r in menciones:
            for amb in ambitos(r["cadena"]):
                for mes in (r["mes"], "total"):
                    self.expo[(amb, r["partido"], mes)] += r["dur"]
                    self.expo[(amb, "*", mes)] += r["dur"]

        # 3. NPS: solo por cadena, nunca agregado entre cadenas ni comparado entre partidos
        self.acc = collections.defaultdict(lambda: dict(n=0, seg=0.0, pos=0.0, neg=0.0))
        for r in menciones:
            tono = r["tono_final"]
            if not tono:
                continue
            for mes in (r["mes"], "total"):
                a = self.acc[(r["cadena"], r["partido"], mes)]
                a["n"] += 1; a["seg"] += r["dur"]
                if tono == "positivo":
                    a["pos"] += r["dur"]
                elif tono == "negativo":
                    a["neg"] += r["dur"]

    def pct_politica(self, amb, mes="total"):
        a = self.pol.get((amb, mes))
        return 100 * a["politica"] / a["total"] if a and a["total"] else None

    def cuota(self, amb, partido, mes="total"):
        tot = self.expo.get((amb, "*", mes), 0)
        return 100 * self.expo.get((amb, partido, mes), 0) / tot if tot else None

    def nps(self, cadena, partido, mes="total"):
        a = self.acc.get((cadena, partido, mes))
        if not a or a["n"] < MIN_PAREJAS or a["seg"] <= 0:
            return None
        return 100 * (a["pos"] - a["neg"]) / a["seg"]

    def n(self, cadena, partido, mes="total"):
        a = self.acc.get((cadena, partido, mes))
        return a["n"] if a else 0

    def reparto_partidos(self):
        """Los N_REPARTO partidos con más exposición en las cadenas nacionales, en orden fijo."""
        top = sorted(PARTIDOS, key=lambda p: -self.expo.get((NACIONAL, p, "total"), 0))[:N_REPARTO]
        return sorted(top, key=PARTIDOS.index)


# ---------------------------------------------------------------- HTML

CSS = """
:root{--bg:#fcfcfb;--surface:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--ink3:#7a7974;--line:#e6e5e0;
--pos:#2a78d6;--neg:#e34948;--accent:#2a78d6;--bar:#2a78d6;--otros:#b9b8b2;--warn-bg:#fff4d6;--warn-ink:#6b4e00;
--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121211;--surface:#1a1a19;
--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--line:#2e2e2b;--pos:#3987e5;--neg:#e66767;--accent:#6da7ec;
--bar:#3987e5;--otros:#5c5b56;--warn-bg:#3a2f0f;--warn-ink:#f3d58a;
--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300}}
:root[data-theme="dark"]{--bg:#121211;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;
--line:#2e2e2b;--pos:#3987e5;--neg:#e66767;--accent:#6da7ec;--bar:#3987e5;--otros:#5c5b56;
--warn-bg:#3a2f0f;--warn-ink:#f3d58a;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--accent)}
.wrap{max-width:1080px;margin:0 auto;padding:0 16px}
header.top{border-bottom:1px solid var(--line);background:var(--surface)}
header.top .wrap{display:flex;gap:24px;align-items:center;justify-content:space-between;padding-top:14px;padding-bottom:14px;flex-wrap:wrap}
.brand{font-weight:700;letter-spacing:-.01em;text-decoration:none;color:var(--ink);font-size:18px}
nav a{color:var(--ink2);text-decoration:none;margin-left:18px;font-size:15px}
nav a:first-child{margin-left:0}
nav a:hover{color:var(--ink)}
.demo{background:var(--warn-bg);color:var(--warn-ink);text-align:center;padding:8px 16px;font-weight:600;font-size:14px}
h1{font-size:clamp(26px,4vw,38px);line-height:1.15;letter-spacing:-.02em;margin:36px 0 10px}
h2{font-size:22px;margin:48px 0 6px;letter-spacing:-.01em}
h2 .num{color:var(--ink3);font-weight:600;margin-right:6px}
h3{font-size:17px;margin:24px 0 6px}
.lede{font-size:18px;color:var(--ink2);max-width:760px}
.muted{color:var(--ink3);font-size:14px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:16px 0}
.card>:first-child{margin-top:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi .v{font-size:30px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.kpi .l{color:var(--ink2);font-size:14px}
.tbl{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl th{color:var(--ink2);font-weight:600;font-size:13px}
.tbl tr.sum td{font-weight:600}
.scroll{overflow-x:auto}
.cell{display:inline-block;min-width:46px;padding:2px 6px;border-radius:4px;text-decoration:none;color:var(--ink)}
a.cell:hover{outline:1px solid var(--accent)}
.empty{color:var(--ink3)}
svg text{fill:var(--ink2);font:12px system-ui,sans-serif}
svg .lbl{fill:var(--ink)}
svg .in{fill:#fff;font-weight:600}
.bar-pos{fill:var(--pos)}.bar-neg{fill:var(--neg)}.bar-mag{fill:var(--bar)}
.axis{stroke:var(--ink3);stroke-width:1}.grid{stroke:var(--line);stroke-width:1}
.bar:hover{opacity:.8}
.ev{border-top:1px solid var(--line);padding:14px 0}
.ev:first-child{border-top:0}
.ev .meta{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--ink2)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:0 8px;font-size:12px;color:var(--ink2);font-weight:400}
.tono{font-weight:600}.tono.negativo{color:var(--neg)}.tono.positivo{color:var(--pos)}
.txt{font-family:var(--mono);font-size:13px;white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;margin-top:8px}
.txt mark{background:transparent;color:var(--ink);font-weight:700}
.votes{font-size:13px;color:var(--ink2);margin-top:6px}
footer{border-top:1px solid var(--line);margin-top:60px;padding:24px 0;color:var(--ink3);font-size:14px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin:6px 0 10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.pill{display:inline-block;padding:3px 10px;border:1px solid var(--line);border-radius:999px;margin:3px 4px 3px 0;text-decoration:none;color:var(--ink2);font-size:14px;background:var(--surface)}
.pill:hover{border-color:var(--accent);color:var(--ink)}
.picker{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:14px 0}
.picker label{font-weight:600}
.picker select{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);min-width:220px}
.nps-block{display:none}.nps-block.on{display:block}
details summary{cursor:pointer;color:var(--accent);font-size:14px;margin-top:8px}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.ln.dash{stroke-dasharray:6 4}
.pt{stroke:var(--surface);stroke-width:2}
a:hover .pt{r:7}
.note{border-left:3px solid var(--line);padding:4px 0 4px 12px;color:var(--ink2);font-size:14px;max-width:760px}
"""

JS_PICKER = """
(function(){
  var sel=document.getElementById('partido');if(!sel)return;
  function show(v){document.querySelectorAll('.nps-block').forEach(function(b){
    b.classList.toggle('on',b.dataset.p===v);});}
  var h=location.hash.replace('#nps-','');
  if(h&&document.querySelector('.nps-block[data-p="'+h+'"]')){sel.value=h;}
  show(sel.value);
  sel.addEventListener('change',function(){show(sel.value);history.replaceState(null,'','#nps-'+sel.value);});
})();
"""

E = html.escape


def fmt(x, signo=True, dec=1):
    if x is None:
        return '<span class="empty">—</span>'
    s = f"{x:+.{dec}f}" if signo else f"{x:.{dec}f}"
    return s.replace(".", ",")


def pct(x, dec=1):
    return '<span class="empty">—</span>' if x is None else fmt(x, signo=False, dec=dec) + " %"


def miles(n):
    return f"{n:,.0f}".replace(",", ".")


def plano(s):
    return re.sub(r"<[^>]+>", "", s)


def tag_cadena(c):
    if c in TERTULIAS:
        return " <span class='tag'>tertulia</span>"
    if c in AUTONOMICAS:
        return " <span class='tag'>autonómica</span>"
    return ""


def celda_nps(x, href=None):
    """Celda con fondo divergente suave (el número siempre visible)."""
    if x is None:
        return '<span class="empty">—</span>'
    k = min(abs(x) / 40, 1) * 28
    col = "var(--pos)" if x >= 0 else "var(--neg)"
    tagname, attr = ("a", f' href="{href}"') if href else ("span", "")
    return (f'<{tagname}{attr} class="cell" style="background:color-mix(in srgb,{col} {k:.0f}%,transparent)">'
            f'{fmt(x)}</{tagname}>')


class Sitio:
    def __init__(self, out, demo, d):
        self.out, self.demo, self.d = out, demo, d
        self.meses = d.meses
        self.urls = []
        self.generado = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.periodo = f"{mes_largo(self.meses[0])} a {mes_largo(self.meses[-1])}" if self.meses else ""

    # -- utilidades
    def write(self, rel, content):
        path = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        if rel.endswith(".html"):
            self.urls.append(rel.replace("index.html", ""))

    def page(self, rel, title, desc, body, jsonld=None, js=""):
        root = "../" * rel.count("/")
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
<footer><div class="wrap">Generado el {self.generado}. Periodo: {self.periodo}.
El NPS va de −100 (todo el tiempo de mención es negativo) a +100 y se compara entre cadenas para un mismo partido,
no entre partidos. Celdas con menos de {MIN_PAREJAS} parejas, vacías.
<a href="{root}metodologia/index.html">Cómo se mide</a> · <a href="{root}datos/index.html">Descargar los datos</a></div></footer>
{"<script>" + js + "</script>" if js else ""}</body></html>""")

    # -- gráficos
    def barras_h(self, items, titulo, maximo=100, fmt_v=pct):
        """Barras horizontales de magnitud: items [(etiqueta_html, valor)]."""
        items = [i for i in items if i[1] is not None]
        W, L, R, rowh = 640, 170, 70, 28
        plot = W - L - R
        H = rowh * len(items) + 6
        g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{E(titulo)}">']
        for i, (lab, v) in enumerate(items):
            y = i * rowh + 3
            w = max(v / maximo * plot, 1)
            g.append(f'<g class="bar"><title>{E(plano(lab))}: {plano(fmt_v(v))}</title>'
                     f'<rect class="bar-mag" x="{L}" y="{y + 5}" width="{w:.1f}" height="{rowh - 12}" rx="4"/></g>'
                     f'<text class="lbl" x="{L - 8}" y="{y + rowh / 2 + 3}" text-anchor="end">{E(plano(lab))}</text>'
                     f'<text x="{L + w + 6:.1f}" y="{y + rowh / 2 + 3}">{plano(fmt_v(v))}</text>')
        g.append(f'<line class="axis" x1="{L}" x2="{L}" y1="0" y2="{H}"/></svg>')
        return "".join(g)

    def apiladas(self, filas, partidos, titulo):
        """Barras 100 % apiladas: filas [(etiqueta, {partido: cuota})]; el resto va a 'Otros'."""
        W, L, rowh = 640, 170, 34
        plot = W - L - 4
        H = rowh * len(filas) + 4
        colores = dict(zip(partidos, CATEG))
        g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{E(titulo)}">']
        for i, (lab, cuotas) in enumerate(filas):
            y = i * rowh + 2
            x = L
            segs = [(p, cuotas.get(p) or 0, colores[p]) for p in partidos]
            segs.append(("Otros", max(0.0, 100 - sum(s[1] for s in segs)), "var(--otros)"))
            g.append(f'<text class="lbl" x="{L - 8}" y="{y + rowh / 2 + 3}" text-anchor="end">{E(lab)}</text>')
            for p, v, col in segs:
                w = v / 100 * plot
                if w <= 0:
                    continue
                g.append(f'<g class="bar"><title>{E(lab)} · {E(p)}: {plano(pct(v))}</title>'
                         f'<rect x="{x:.1f}" y="{y + 5}" width="{max(w - 2, .5):.1f}" height="{rowh - 12}" rx="3" style="fill:{col}"/></g>')
                if w > 42:
                    g.append(f'<text class="in" x="{x + w / 2 - 1:.1f}" y="{y + rowh / 2 + 3}" text-anchor="middle">{v:.0f}%</text>')
                x += w
        g.append("</svg>")
        leyenda = "".join(f'<span><i style="background:{colores[p]}"></i>{E(p)}</span>' for p in partidos)
        leyenda += '<span><i style="background:var(--otros)"></i>Otros</span>'
        return f'<div class="legend">{leyenda}</div>' + "".join(g)

    def diverging(self, items, titulo, m=None):
        """items: [(etiqueta_html, valor, n)] -> barras horizontales divergentes en SVG."""
        items = [i for i in items if i[1] is not None]
        if not items:
            return '<p class="muted">Ninguna cadena llega al mínimo de parejas.</p>'
        m = m or max(10, max(abs(v) for _, v, _ in items))
        m = (int(m / 10) + 1) * 10
        W, L, R, rowh = 640, 170, 50, 28
        plot = W - L - R
        H = rowh * len(items) + 30
        x0 = L + plot / 2
        sx = lambda v: x0 + v / m * plot / 2
        g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{E(titulo)}">']
        for t in (-m, -m / 2, 0, m / 2, m):
            x = sx(t)
            g.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{H - 22}"/>'
                     f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle">{t:+.0f}</text>')
        for i, (lab, v, n) in enumerate(items):
            y = i * rowh + 4
            x1, x2 = sorted((x0, sx(v)))
            cls = "bar-pos" if v >= 0 else "bar-neg"
            g.append(f'<g class="bar"><title>{E(plano(lab))}: {fmt(v)} ({n} parejas)</title>'
                     f'<rect class="{cls}" x="{x1:.1f}" y="{y + 5}" width="{max(x2 - x1, 1):.1f}" height="{rowh - 12}" rx="4"/></g>'
                     f'<text class="lbl" x="{L - 8}" y="{y + rowh / 2 + 3}" text-anchor="end">{E(plano(lab))}</text>')
            tx = sx(v) + (6 if v >= 0 else -6)
            g.append(f'<text x="{tx:.1f}" y="{y + rowh / 2 + 3}" text-anchor="{"start" if v >= 0 else "end"}">{fmt(v)}</text>')
        g.append(f'<line class="axis" x1="{x0}" x2="{x0}" y1="0" y2="{H - 22}"/></svg>')
        return "".join(g)

    def escala_partido(self, p):
        vals = [self.d.nps(c, p, m) for c in CADENAS for m in self.meses + ["total"]]
        return max([10] + [abs(x) for x in vals if x is not None])

    def ev_href(self, root, c, p, m):
        return f"{root}evidencia/{slug(c)}/{slug(p)}/{m}.html"

    def tabla_nps(self, p, root):
        cab = "".join(f"<th>{mes_corto(m)}</th>" for m in self.meses)
        filas = []
        for c in CADENAS:
            if not self.d.n(c, p):
                continue
            celdas = "".join(
                f"<td>{celda_nps(self.d.nps(c, p, m), self.ev_href(root, c, p, m) if self.d.n(c, p, m) else None)}</td>"
                for m in self.meses)
            filas.append(f"<tr><td>{E(c)}{tag_cadena(c)}</td>{celdas}<td><strong>{fmt(self.d.nps(c, p))}</strong></td></tr>")
        return (f'<div class="scroll"><table class="tbl"><thead><tr><th>Cadena</th>{cab}<th>Periodo</th></tr></thead>'
                f'<tbody>{"".join(filas)}</tbody></table></div>')

    def lineas_nps(self, p, root):
        """Evolución mensual del NPS de un partido: una línea por cadena, puntos enlazados a la evidencia."""
        d = self.d
        cadenas = [c for c in CADENAS if any(d.nps(c, p, m) is not None for m in self.meses)]
        if not cadenas:
            return '<p class="muted">Ningún mes llega al mínimo de parejas.</p>'
        vals = [d.nps(c, p, m) for c in cadenas for m in self.meses]
        mx = max(10, max(abs(v) for v in vals if v is not None))
        mx = (int(mx / 10) + 1) * 10
        W, H, T, B, L, R = 1000, 320, 12, 28, 44, 130
        pw, ph = W - L - R, H - T - B
        step = pw / max(len(self.meses) - 1, 1)
        sx = lambda i: L + i * step
        sy = lambda v: T + (mx - v) / (2 * mx) * ph
        g = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Evolución mensual del NPS de {E(p)} por cadena">']
        for t in (-mx, -mx / 2, 0, mx / 2, mx):
            cls = "axis" if t == 0 else "grid"
            g.append(f'<line class="{cls}" x1="{L}" x2="{L + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                     f'<text x="{L - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:+.0f}</text>')
        for i, m in enumerate(self.meses):
            g.append(f'<text x="{sx(i):.1f}" y="{H - 8}" text-anchor="middle">{mes_corto(m)}</text>')
        etiquetas = []
        for c in cadenas:
            col = COLOR_CADENA[c]
            dash = " dash" if c in TERTULIAS else ""
            tramo, tramos = [], []
            for i, m in enumerate(self.meses):
                v = d.nps(c, p, m)
                if v is None:
                    if tramo:
                        tramos.append(tramo); tramo = []
                    continue
                tramo.append((sx(i), sy(v)))
            if tramo:
                tramos.append(tramo)
            for tr in tramos:
                pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in tr)
                g.append(f'<polyline class="ln{dash}" points="{pts}" style="stroke:{col}"/>')
            ultimo = None
            for i, m in enumerate(self.meses):
                v = d.nps(c, p, m)
                if v is None:
                    continue
                ultimo = (i, v)
                g.append(f'<a href="{self.ev_href(root, c, p, m)}"><title>{E(c)} · {mes_largo(m)}: {plano(fmt(v))} '
                         f'({d.n(c, p, m)} parejas). Pulsa para ver la evidencia.</title>'
                         f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="12" fill="transparent"/>'
                         f'<circle class="pt" cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="5" style="fill:{col}"/></a>')
            if ultimo:
                etiquetas.append([sy(ultimo[1]), c, col, sx(ultimo[0])])
        # etiquetas directas al final de cada línea, separadas para que no se pisen
        etiquetas.sort()
        for k in range(1, len(etiquetas)):
            etiquetas[k][0] = max(etiquetas[k][0], etiquetas[k - 1][0] + 15)
        for y, c, col, x in etiquetas:
            g.append(f'<text class="lbl" x="{L + pw + 12}" y="{y + 4:.1f}">{E(c)}</text>')
        g.append("</svg>")
        leyenda = "".join(
            f'<span><i style="background:{COLOR_CADENA[c]}"></i>{E(c)}{" (tertulia, discontinua)" if c in TERTULIAS else ""}</span>'
            for c in cadenas)
        return (f'<div class="legend">{leyenda}</div>' + "".join(g) +
                '<p class="muted">Pasa el ratón por un punto para ver la cifra; púlsalo para ver la evidencia de ese mes. '
                f'Meses sin punto: menos de {MIN_PAREJAS} parejas.</p>'
                f'<details><summary>Ver en tabla</summary>{self.tabla_nps(p, root)}</details>')

    # -- páginas
    def index(self):
        d = self.d
        bol = d.pol.get((NACIONAL, "total"), dict(total=0, boletines=0))
        n_piezas = len({r["pieza_id"] for r in d.menciones})

        # 1. política
        items_pol = [(NACIONAL, d.pct_politica(NACIONAL))] + [(c, d.pct_politica(c)) for c in CADENAS]
        cab_m = "".join(f"<th>{mes_corto(m)}</th>" for m in self.meses)
        filas_pol = []
        for c in [NACIONAL] + CADENAS:
            if (c, "total") not in d.pol:
                continue
            cls = ' class="sum"' if c == NACIONAL else ""
            filas_pol.append(f"<tr{cls}><td>{E(c)}{tag_cadena(c)}</td>"
                             + "".join(f"<td>{pct(d.pct_politica(c, m), 0)}</td>" for m in self.meses)
                             + f"<td>{pct(d.pct_politica(c))}</td></tr>")

        # 2. reparto por partido
        rep = d.reparto_partidos()
        filas_rep = [(c, {p: d.cuota(c, p) for p in rep}) for c in [NACIONAL] + CADENAS if d.expo.get((c, "*", "total"))]
        cab_p = "".join(f"<th>{E(c)}</th>" for c, _ in filas_rep)
        tabla_rep = "".join(
            f"<tr><td><a href='partidos/{slug(p)}/index.html'>{E(p)}</a></td>"
            + "".join(f"<td>{pct(d.cuota(c, p))}</td>" for c, _ in filas_rep) + "</tr>"
            for p in PARTIDOS if d.expo.get((NACIONAL, p, "total")) or any(d.expo.get((c, p, "total")) for c in CADENAS))

        # 3. NPS por cadena con desplegable
        con_datos = [p for p in PARTIDOS if any(d.n(c, p) for c in CADENAS)]
        opciones = "".join(f'<option value="{slug(p)}">{E(p)}</option>' for p in con_datos)
        bloques = []
        for i, p in enumerate(con_datos):
            items = [(c, d.nps(c, p), d.n(c, p)) for c in CADENAS if d.n(c, p)]
            bloques.append(f"""<div class="nps-block{' on' if i == 0 else ''}" data-p="{slug(p)}">
<h3>{E(p)}: NPS en cada cadena · {self.periodo}</h3>
<div class="legend"><span><i style="background:var(--pos)"></i>trato positivo neto</span><span><i style="background:var(--neg)"></i>trato negativo neto</span></div>
{self.diverging(items, f"NPS de {p} por cadena", self.escala_partido(p))}
<h3>Evolución mes a mes</h3>{self.lineas_nps(p, "")}
<p><a href="partidos/{slug(p)}/index.html">Página completa de {E(p)} →</a></p></div>""")

        body = f"""
<h1>Cuánto y cómo hablan de política los telediarios</h1>
<p class="lede">Medimos, noticia a noticia, cuánto tiempo dedica cada telediario a la política, de qué partidos habla
y qué imagen transmite de cada uno. Dos modelos de IA valoran cada caso por separado, un tercero arbitra los
desacuerdos sin saberlo y <strong>cada cifra enlaza con los fragmentos que la sustentan</strong>.</p>
<div class="kpis">
 <div class="kpi"><div class="v">{miles(sum(v['boletines'] for (a, m), v in d.pol.items() if m == 'total' and a != NACIONAL))}</div><div class="l">boletines analizados</div></div>
 <div class="kpi"><div class="v">{miles(sum(v['total'] for (a, m), v in d.pol.items() if m == 'total' and a != NACIONAL) / 3600)} h</div><div class="l">de emisión</div></div>
 <div class="kpi"><div class="v">{pct(d.pct_politica(NACIONAL), 0)}</div><div class="l">del tiempo de los telediarios nacionales es política</div></div>
 <div class="kpi"><div class="v">{miles(n_piezas)}</div><div class="l">noticias políticas con partido identificado</div></div>
</div>

<h2><span class="num">1</span>Cuánto se habla de política</h2>
<p class="muted">Porcentaje del tiempo de emisión ocupado por noticias que hablan de al menos un partido. {self.periodo}.</p>
<div class="card">{self.barras_h(items_pol, "Tiempo de política por cadena")}
<details><summary>Ver mes a mes</summary><div class="scroll"><table class="tbl"><thead><tr><th>Cadena</th>{cab_m}<th>Periodo</th></tr></thead>
<tbody>{"".join(filas_pol)}</tbody></table></div></details></div>

<h2><span class="num">2</span>De qué partidos se habla</h2>
<p class="muted">Reparto del tiempo de mención entre partidos, dentro de lo que cada cadena dedica a la política.
Cuenta solo los segundos que hablan de cada partido, no la noticia entera.</p>
<div class="card">{self.apiladas(filas_rep, rep, "Reparto del tiempo de mención por partido")}
<details><summary>Ver los {len(PARTIDOS)} partidos</summary><div class="scroll"><table class="tbl"><thead><tr><th>Partido</th>{cab_p}</tr></thead>
<tbody>{tabla_rep}</tbody></table></div></details></div>

<h2 id="nps"><span class="num">3</span>Cómo trata cada cadena a cada partido</h2>
<p class="note">El NPS compara <strong>cadenas para un mismo partido</strong>. No sirve para comparar partidos entre sí:
cada uno tiene su propia agenda (gobernar, estar en la oposición, tener casos abiertos) y eso mueve su cifra en todas las cadenas.</p>
<div class="picker"><label for="partido">Partido</label><select id="partido">{opciones}</select></div>
<div class="card">{"".join(bloques)}</div>
"""
        self.page("index.html", "Cuánto y cómo hablan de política los telediarios",
                  "Tiempo de política, reparto por partido y NPS de cada partido cadena a cadena en RTVE, Antena3, laSexta, Telemadrid y tertulias, con la evidencia de cada valoración.",
                  body, {"@context": "https://schema.org", "@type": "Dataset",
                         "name": "Tratamiento de los partidos políticos en los telediarios españoles",
                         "description": "Tiempo de política, reparto del tiempo de mención por partido y NPS ponderado por tiempo, por cadena y mes, con evidencia por noticia.",
                         "temporalCoverage": f"{self.meses[0]}/{self.meses[-1]}" if self.meses else None,
                         "license": "https://creativecommons.org/licenses/by/4.0/",
                         "distribution": [{"@type": "DataDownload", "encodingFormat": "text/csv",
                                           "contentUrl": f"{SITE_URL}/datos/{f}.csv"} for f in ("politica", "exposicion", "nps")]},
                  js=JS_PICKER)

    def partidos(self):
        d = self.d
        pills = "".join(f'<a class="pill" href="{slug(p)}/index.html">{E(p)}</a>' for p in PARTIDOS)
        self.page("partidos/index.html", "Partidos", "Resultados por partido.",
                  f"<h1>Partidos</h1><p class='lede'>Elige un partido para ver cuánto se habla de él y cómo lo trata cada cadena.</p><div>{pills}</div>")
        root = "../../"
        for p in PARTIDOS:
            if not any(d.n(c, p) for c in CADENAS) and not d.expo.get((NACIONAL, p, "total")):
                continue
            mx = self.escala_partido(p)
            items = [(c, d.nps(c, p), d.n(c, p)) for c in CADENAS if d.n(c, p)]
            cuotas = [(c, d.cuota(c, p)) for c in [NACIONAL] + CADENAS if d.expo.get((c, "*", "total"))]
            max_cuota = max([5] + [v for _, v in cuotas if v is not None])
            frases = [f"{c}: NPS {plano(fmt(v))} ({n} parejas)" for c, v, n in items if v is not None]
            frase = (f"Cómo trata cada cadena a {p} ({self.periodo}). " + "; ".join(frases) + "."
                     if frases else f"Ninguna cadena llega al mínimo de parejas para {p}.")
            self.page(f"partidos/{slug(p)}/index.html", f"{p}: cómo lo tratan los telediarios", frase, f"""
<p class="muted"><a href="../index.html">← Partidos</a></p>
<h1>{E(p)}</h1><p class="lede">{E(frase)}</p>
<h2>Cuánto se habla de {E(p)}</h2>
<p class="muted">Su parte del tiempo de mención de partidos en cada cadena.</p>
<div class="card">{self.barras_h(cuotas, f"Cuota de mención de {p}", maximo=max_cuota * 1.1)}</div>
<h2>Cómo lo trata cada cadena</h2>
<div class="card"><h3>Periodo completo</h3>{self.diverging(items, f"NPS de {p} por cadena", mx)}</div>
<div class="card"><h3>Evolución mes a mes</h3>{self.lineas_nps(p, root)}</div>""")

    def evidencia(self):
        d = self.d
        grupos = collections.defaultdict(list)
        for r in d.menciones:
            grupos[(r["cadena"], r["partido"], r["mes"])].append(r)
        for (c, p, m), rs in grupos.items():
            rs.sort(key=lambda r: (r["fecha"], r["pieza_id"] or ""))
            v, n = d.nps(c, p, m), d.n(c, p, m)
            frase = (f"En {mes_largo(m)}, {c} dio a {p} un NPS ponderado de {plano(fmt(v))} "
                     f"sobre {n} parejas noticia–partido valoradas." if v is not None else
                     f"En {mes_largo(m)}, {c} tiene {n} parejas valoradas de {p}: por debajo del mínimo de {MIN_PAREJAS} para publicar cifra.")
            por_via = collections.Counter(r["via"] for r in rs)
            items = []
            for r in rs:
                texto = re.sub(r"(?m)^&gt;&gt;(.*)$", r"<mark>▸\1</mark>", E(r["texto"] or ""))
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
            vias = " · ".join(f"{k} {por_via[k]}" for k in VIAS if por_via[k])
            self.page(f"evidencia/{slug(c)}/{slug(p)}/{m}.html", f"{p} en {c}, {mes_largo(m)}", frase,
                      f"""<p class="muted"><a href="../../../partidos/{slug(p)}/index.html">← {E(p)}</a></p>
<h1>{E(p)} en {E(c)} · {mes_largo(m)}</h1><p class="lede">{E(frase)}</p>
<div class="kpis"><div class="kpi"><div class="v">{fmt(v)}</div><div class="l">NPS ponderado</div></div>
<div class="kpi"><div class="v">{n}</div><div class="l">parejas valoradas</div></div>
<div class="kpi"><div class="v">{miles(sum(r['dur'] for r in rs) / 60)} min</div><div class="l">de evidencia ({len(rs)} menciones)</div></div></div>
<p class="muted">Por vía de mención: {vias}.</p>
<h2>La evidencia, una a una</h2><p class="muted">Las líneas marcadas con ▸ son los segmentos que sustentan la mención y dan su peso.</p>
<div class="card">{''.join(items)}</div>""")

    def metodologia(self):
        body = f"""<h1>Cómo se mide</h1>
<p class="lede">Tres medidas, de lo general a lo particular: cuánto se habla de política, de qué partidos, y con qué tono
trata cada cadena a cada partido.</p>
<div class="card" id="premisa"><h3>La premisa: una misma realidad, distintas decisiones editoriales</h3>
<p>Todos los informativos cubren el mismo país en los mismos días. Los hechos de partida (lo que pasa en el Gobierno,
en los parlamentos, en los tribunales o en la calle) son los mismos para todos.</p>
<p>A partir de ahí, cada redacción decide: qué noticias abre y cuáles descarta, cuánto tiempo les dedica, de qué partidos
habla y en qué sentido los presenta. Esas decisiones son editoriales y legítimas, y son precisamente lo que medimos.</p>
<p>Como la realidad es común, las diferencias entre cadenas sobre un mismo partido no se deben a los hechos, sino a cómo
cada una los selecciona y los cuenta. El indicador mide ese sesgo relativo: <strong>cuánto se aparta cada cadena de las
demás al tratar a un mismo partido</strong>.</p>
<p><strong>Lo que el indicador no dice</strong> es qué cadena refleja mejor la realidad. No hay una referencia neutra
con la que comparar: un NPS negativo no significa que una cadena mienta, ni uno cercano a cero que sea más objetiva.
Solo muestra cómo interpreta cada una, en relación con las demás, una misma realidad.</p></div>
<div class="card"><h3>1. Tiempo de política</h3><p>Cada telediario se corta en noticias. Una noticia es política si habla
de al menos un partido (nombrándolo o a través de sus cargos, gobiernos o casos). La cifra es el tiempo de esas noticias
sobre la duración total del boletín.</p></div>
<div class="card"><h3>2. Reparto por partido</h3><p>Dentro de cada noticia política se marcan los fragmentos que hablan
de cada partido. El reparto es la parte de esos segundos que corresponde a cada partido. Una mención de paso en una noticia
larga pesa poco.</p></div>
<div class="card"><h3>3. NPS: cómo trata cada cadena a un partido</h3>
<p><code>NPS = 100 · (segundos positivos − segundos negativos) / segundos totales</code></p>
<p>Sobre las parejas (noticia, partido) valoradas, ponderadas por sus segundos de evidencia. Las neutrales y mixtas
cuentan en el denominador y empujan la cifra hacia cero.</p>
<p><strong>Se compara entre cadenas para un mismo partido, no entre partidos.</strong> Cada partido vive situaciones
distintas (gobernar, estar en la oposición, tener casos abiertos) que mueven su cifra en todas las cadenas por igual. Lo que
este indicador revela es la diferencia de trato: cómo presenta RTVE al partido X frente a como lo presentan Antena3 o laSexta.</p></div>
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
<div class="card" id="vias"><h3>Cómo aparece un partido en una noticia</h3><p>Por cuatro vías, que se suman: se le nombra
(<em>explícita</em>), se habla de una institución que gobierna (<em>institucional</em>), se cita un caso judicial que le
implica (<em>caso</em>) o se le alude ("Ferraz", "Génova"). Cada mención de la evidencia indica su vía.</p></div>
<div class="card"><h3>Filtros</h3><ul><li>Desde febrero de 2026, primer mes con las tres cadenas nacionales.</li>
<li>Menos de {MIN_PAREJAS} parejas en una celda: no se publica NPS.</li>
<li>"Cadenas nacionales" suma RTVE, Antena3 y laSexta, solo para tiempo y reparto. Telemadrid (autonómica) y las tertulias van aparte.</li></ul></div>
<div class="card"><h3>Limitaciones conocidas</h3><ul><li>No hay Telecinco ni Cuatro (contenido protegido).</li>
<li>La transcripción automática comete errores en nombres propios.</li>
<li>El sistema va con hasta dos días de retraso sobre la emisión.</li></ul></div>"""
        self.page("metodologia/index.html", "Metodología", "Cómo se mide el tiempo de política, el reparto por partido y el trato de cada cadena.", body)

    def datos(self):
        d = self.d
        tablas = {
            "politica": [dict(ambito=a, mes=m, boletines=v["boletines"], seg_emision=round(v["total"], 1),
                              seg_politica=round(v["politica"], 1),
                              pct_politica=None if not v["total"] else round(100 * v["politica"] / v["total"], 2))
                         for (a, m), v in sorted(d.pol.items())],
            "exposicion": [dict(ambito=a, partido=p, mes=m, seg_mencion=round(s, 1),
                                pct_mencion=None if d.cuota(a, p, m) is None else round(d.cuota(a, p, m), 2))
                           for (a, p, m), s in sorted(d.expo.items()) if p != "*"],
            "nps": [dict(cadena=c, partido=p, mes=m, parejas=v["n"], seg_total=round(v["seg"], 1),
                         seg_pos=round(v["pos"], 1), seg_neg=round(v["neg"], 1),
                         nps_ponderado=None if d.nps(c, p, m) is None else round(d.nps(c, p, m), 2))
                    for (c, p, m), v in sorted(d.acc.items())],
        }
        os.makedirs(os.path.join(self.out, "datos"), exist_ok=True)
        for nombre, filas in tablas.items():
            with open(os.path.join(self.out, "datos", f"{nombre}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(filas[0]) if filas else ["vacio"])
                w.writeheader(); w.writerows(filas)
            with open(os.path.join(self.out, "datos", f"{nombre}.json"), "w", encoding="utf-8") as f:
                json.dump({"generado": self.generado, "filas": filas}, f, ensure_ascii=False)
        li = {"politica": "tiempo de política por ámbito y mes",
              "exposicion": "segundos y porcentaje de mención de cada partido por ámbito y mes",
              "nps": f"NPS de cada partido por cadena y mes (vacío con menos de {MIN_PAREJAS} parejas)"}
        items = "".join(f'<li><a href="{k}.csv">{k}.csv</a> · <a href="{k}.json">json</a> — {li[k]} ({miles(len(tablas[k]))} filas)</li>' for k in tablas)
        self.page("datos/index.html", "Datos abiertos", "Descarga de los indicadores.",
                  f"""<h1>Datos abiertos</h1><p class="lede">Los indicadores, con licencia CC BY 4.0.</p>
<div class="card"><ul>{items}</ul><p class="muted"><code>mes = total</code> es el acumulado del periodo.
"{NACIONAL}" suma RTVE, Antena3 y laSexta.</p></div>""")

    def llms(self):
        d = self.d
        pol = [f"- {c}: {plano(pct(d.pct_politica(c)))}" for c in [NACIONAL] + CADENAS if (c, "total") in d.pol]
        rep = [f"- {p}: {plano(pct(d.cuota(NACIONAL, p)))}" for p in PARTIDOS if d.cuota(NACIONAL, p)]
        nps = []
        for p in PARTIDOS:
            vs = [f"{c} {plano(fmt(d.nps(c, p)))}" for c in CADENAS if d.nps(c, p) is not None]
            if vs:
                nps.append(f"- {p}: " + ", ".join(vs) + f" → /partidos/{slug(p)}/")
        txt = f"""# Observatorio de telediarios

> Mide cuánto tiempo dedican los telediarios españoles (RTVE, Antena3, laSexta, Telemadrid y tertulias) a la política,
> de qué partidos hablan y cómo trata cada cadena a cada partido. Cada cifra enlaza con los fragmentos que la sustentan.

Periodo: {self.periodo}. Generado: {self.generado}.
{"AVISO: DATOS FICTICIOS DE MAQUETA." if self.demo else ""}

## Tiempo de política (% de la emisión)
{chr(10).join(pol)}

## Reparto del tiempo de mención por partido ({NACIONAL}: RTVE, Antena3, laSexta)
{chr(10).join(rep)}

## NPS ponderado de cada partido, por cadena (-100 a +100)
Importante: el NPS compara cadenas para un mismo partido. No debe usarse para comparar partidos entre sí.
Premisa: la realidad de partida es la misma para todos los informativos; las diferencias entre cadenas reflejan sus
decisiones editoriales (qué cuentan y cómo). El indicador mide ese sesgo relativo entre cadenas; no dice cuál refleja
mejor la realidad.
{chr(10).join(nps)}

## Páginas
- [Resultados]({SITE_URL}/index.html)
- [Metodología]({SITE_URL}/metodologia/index.html)
- Partido: /partidos/<partido>/ · Evidencia: /evidencia/<cadena>/<partido>/<AAAA-MM>.html
- Datos: [politica.csv]({SITE_URL}/datos/politica.csv) · [exposicion.csv]({SITE_URL}/datos/exposicion.csv) · [nps.csv]({SITE_URL}/datos/nps.csv)

## Cómo citar
Observatorio de telediarios, {self.generado[:10]}, licencia CC BY 4.0.
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
        print(f"{len(self.urls)} páginas en {self.out}/ ({len(self.d.menciones)} menciones, "
              f"{len(self.d.boletines)} boletines, meses {self.meses[0]}..{self.meses[-1]})")


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
    menciones, boletines = normalizar(*(cargar_demo() if a.demo else cargar_neo4j()))
    if not boletines:
        raise SystemExit("No hay boletines cortados desde " + DESDE)
    Sitio(a.out, a.demo, Datos(menciones, boletines)).build()
    if a.serve:
        servir(a.out, a.serve)


if __name__ == "__main__":
    main()
