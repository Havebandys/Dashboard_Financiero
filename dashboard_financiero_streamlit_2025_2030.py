import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==========================
# Configuración general
# ==========================
st.set_page_config(
    page_title="📊 Dashboard Financiero - Cobertura de Gastos",
    page_icon="💼",
    layout="wide",
)

# Meses en español en orden BI
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
MESES_IDX = {m: i for i, m in enumerate(MESES, start=1)}

NUMERIC_COLS_RESUMEN = [
    "Total Ingresos",
    "Total Gastos",
    "Rentabilidad Inversiones",
    "Rentabilidad Total",
    "Rentabilidad Acumulada",
    "Objetivo",
]

REQUIRED_SHEETS = {
    "Inversiones": ["Mes", "Año", "Activo", "Monto Invertido", "Rentabilidad (%)", "Valor Actual", "Rentabilidad Real"],
    "Ingresos Profesionales": ["Mes", "Año", "Concepto", "Monto"],
    "Gastos Fijos": ["Mes", "Año", "Concepto", "Monto"],
    "Resumen Mensual": [
        "Mes", "Año",
        "Total Ingresos", "Total Gastos",
        "Rentabilidad Inversiones", "Rentabilidad Total",
        "Rentabilidad Acumulada", "Objetivo",
    ],
}

# ==========================
# Utilidades
# ==========================
@st.cache_data(show_spinner=False)
def contrast_text_colors(hex_list):
    """Devuelve #000 o #FFF por slice según luminancia percibida."""
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    out = []
    for hx in hex_list:
        r, g, b = hex_to_rgb(hx)
        y = 0.2126*r + 0.7152*g + 0.0722*b  # luminancia perceptual
        out.append("#000000" if y > 186 else "#FFFFFF")
    return out


def cargar_excel(content_bytes: bytes):
    """Carga todas las hojas requeridas desde un archivo Excel (contenido en bytes)."""
    xls = pd.ExcelFile(io.BytesIO(content_bytes))
    hojas = {}
    for sheet, cols in REQUIRED_SHEETS.items():
        if sheet not in xls.sheet_names:
            raise ValueError(f"Falta la hoja '{sheet}' en el Excel.")
        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        faltantes = [c for c in cols if c not in df.columns]
        if faltantes:
            raise ValueError(f"La hoja '{sheet}' no tiene estas columnas requeridas: {faltantes}")
        hojas[sheet] = df
    return hojas


def ordenar_meses(df: pd.DataFrame) -> pd.DataFrame:
    if "Mes" in df.columns:
        cat = pd.api.types.CategoricalDtype(categories=MESES, ordered=True)
        df["Mes"] = df["Mes"].astype(cat)
        df = df.sort_values(["Año", "Mes"]).reset_index(drop=True)
    return df


def completar_meses_ano(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Garantiza 12 meses presentes para el año seleccionado, rellenando con 0 donde falte."""
    base = pd.DataFrame({"Año": [year] * 12, "Mes": MESES})
    out = base.merge(df, on=["Año", "Mes"], how="left")
    for col in NUMERIC_COLS_RESUMEN:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return ordenar_meses(out)


def moneda(num: float) -> str:
    try:
        return f"$ {num:,.0f}".replace(",", ".")
    except Exception:
        return str(num)


def theme_center_color() -> str:
    try:
        base = st.get_option("theme.base")
    except Exception:
        base = None
    return "rgba(0,0,0,0.85)" if base == "light" else "rgba(255,255,255,0.95)"


def meses_desde(m_ini_idx: int, n_meses: int) -> list[int]:
    """Devuelve lista de índices de meses (1..12) desde m_ini_idx, con wrap."""
    seq = []
    cur = m_ini_idx
    for _ in range(n_meses):
        seq.append(cur)
        cur = 1 if cur == 12 else cur + 1
    return seq


def objetivos_para_horizonte(resumen_all: pd.DataFrame, año_ini: int, m_ini_idx: int, H: int) -> list[float]:
    """Lista de objetivos para H meses a partir de (año_ini, m_ini_idx), cruzando año si hace falta."""
    obj = []
    y = año_ini
    mis = meses_desde(m_ini_idx, H)
    i = 0
    while i < H:
        m_idx = mis[i]
        mes_nombre = MESES[m_idx - 1]
        fila = resumen_all[(resumen_all["Año"] == y) & (resumen_all["Mes"] == mes_nombre)]
        obj.append(float(fila["Objetivo"].values[0]) if not fila.empty else 0.0)
        if m_idx == 12 and (i + 1) < H:
            y += 1
        i += 1
    return obj


def muestra_neta_para_mes_inicio(resumen_all: pd.DataFrame, m_ini_idx: int, H: int) -> np.ndarray:
    """
    Pool de netos mensuales (Ingresos + RentInv - Gastos) para bootstrap,
    usando los meses del horizonte [m_ini_idx .. m_ini_idx+H-1] sobre TODOS los años.
    """
    meses_set = set(meses_desde(m_ini_idx, H))
    df = resumen_all.copy()
    df["midx"] = df["Mes"].map(MESES_IDX).astype(int)
    df["net"] = df["Total Ingresos"] + df["Rentabilidad Inversiones"] - df["Total Gastos"]
    pool = df[df["midx"].isin(meses_set)]["net"].to_numpy(dtype=float)
    pool = np.nan_to_num(pool, nan=0.0)
    return pool


# ==========================
# Sidebar - Carga y controles
# ==========================
st.sidebar.header("📂 Cargar datos")
archivo = st.sidebar.file_uploader("Subí el Excel con tus datos (xlsx)", type=["xlsx"])

años_por_defecto = list(range(2025, 2031))

# Fecha objetivo (opcional)
use_date_obj = st.sidebar.toggle("📆 Usar fecha objetivo", value=False)
fecha_objetivo = None
if use_date_obj:
    fecha_objetivo = st.sidebar.date_input("Fecha objetivo", value=date.today())

# Controles adicionales
mostrar_tabla = st.sidebar.toggle("📋 Mostrar tabla mensual", value=False)

# Galería de donuts por año
ver_galeria = st.sidebar.toggle("🧭 Ver donuts por año (2025–2030)", value=False)
cols_galeria = st.sidebar.slider("Columnas galería", 2, 6, 4) if ver_galeria else None

# Monte Carlo
st.sidebar.header("🔮 Proyección")
mes_inicio_mc = st.sidebar.selectbox("Mes de inicio proyección", MESES, index=0)
num_sims = int(st.sidebar.number_input("Simulaciones (MC)", min_value=1000, max_value=200000, value=5000, step=1000))
metodo_mc = st.sidebar.selectbox("Método", ["Bootstrap (empírico)", "Normal (μ, σ)"])
semilla = int(st.sidebar.number_input("Semilla", min_value=0, value=42, step=1))

# Semáforo (umbral amarillo)
umbral_amarillo = int(st.sidebar.slider("Umbral amarillo (≥ % cobertura)", 50, 99, 85))

def semaforo_color(pct: float, y_floor: int = 85) -> str:
    if pct >= 100:
        return "🟢"
    elif pct >= y_floor:
        return "🟡"
    return "🔴"


# ==========================
# Título y guía
# ==========================
st.title("💼 Dashboard Financiero · Cobertura de gastos comprometidos")
st.caption(
    """Donut = Cobertura de gastos **anual**: (Ingresos + Rentabilidad de Inversiones) / Objetivo (Gastos).
La proyección estima la **probabilidad** de cubrir gastos en 3, 6 y 12 meses bajo Monte Carlo."""
)

if not archivo:
    st.info(
        """Subí un Excel con las hojas **Inversiones**, **Ingresos Profesionales**, **Gastos Fijos** y **Resumen Mensual**.
Podés usar el modelo 2025–2030."""
    )
    st.stop()

# ==========================
# Carga segura del Excel
# ==========================
try:
    hojas = cargar_excel(archivo.getvalue())
except Exception as e:
    st.error(f"No se pudo leer el Excel: {e}")
    st.stop()

resumen_all: pd.DataFrame = hojas["Resumen Mensual"].copy()
resumen_all = ordenar_meses(resumen_all)
for col in NUMERIC_COLS_RESUMEN:
    resumen_all[col] = pd.to_numeric(resumen_all[col], errors="coerce").fillna(0)

# Años disponibles y selector
años_disponibles = sorted(set(resumen_all["Año"]))
años_filtrables = [a for a in años_disponibles if 2025 <= int(a) <= 2030] or años_por_defecto

col_year, _ = st.columns([1, 3])
with col_year:
    año_sel = st.selectbox("Año", años_filtrables, index=0)

# ==========================
# Filtrado anual y métricas
# ==========================
resumen_y = resumen_all[resumen_all["Año"] == año_sel].copy()
resumen_y = completar_meses_ano(resumen_y, año_sel)

# Agregados anuales
ingresos_anual = float(resumen_y["Total Ingresos"].sum())
gastos_anual = float(resumen_y["Total Gastos"].sum())
rent_inv_anual = float(resumen_y["Rentabilidad Inversiones"].sum())
objetivo_anual = float(resumen_y["Objetivo"].sum())  # por modelo: = gastos

cubierto_monto = ingresos_anual + rent_inv_anual
faltante_monto = max(objetivo_anual - cubierto_monto, 0.0)
excedente_monto = max(cubierto_monto - objetivo_anual, 0.0)

cobertura_pct = 0.0 if objetivo_anual <= 0 else (cubierto_monto / objetivo_anual) * 100
center_col = theme_center_color()

# Neto mensual
resumen_y["Neto"] = resumen_y["Total Ingresos"] + resumen_y["Rentabilidad Inversiones"] - resumen_y["Total Gastos"]
avg_obj_mensual = float(resumen_y["Objetivo"].mean()) if len(resumen_y) else 0.0

# ==========================
# KPIs
# ==========================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Ingresos (año)", moneda(ingresos_anual))
col2.metric("🏠 Gastos (año)", moneda(gastos_anual))
col3.metric("📈 Rentab. Inversiones (año)", moneda(rent_inv_anual))
col4.metric("🎯 Objetivo (año)", moneda(objetivo_anual))
if excedente_monto > 0:
    col5.metric("✅ Excedente vs objetivo", moneda(excedente_monto))
else:
    col5.metric("⏳ Faltante vs objetivo", moneda(faltante_monto))

if use_date_obj and fecha_objetivo:
    dias_restantes = (fecha_objetivo - date.today()).days
    st.metric("⏱️ Días restantes al objetivo", f"{dias_restantes} días")

# ==========================
# Donut principal (etiquetas dentro + año al centro + semáforo)
# ==========================

covered = min(cubierto_monto, objetivo_anual)
faltante = max(objetivo_anual - covered, 0.0)
sem_icon = semaforo_color(cobertura_pct, umbral_amarillo)

# colores de slices + auto-contraste del texto
slice_colors = ["#2ECC71", "#B04C5A"]  # verde / bordó claro
def _contrast(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if y > 186 else "#FFFFFF"
text_colors = [_contrast(c) for c in slice_colors]

fig_donut = go.Figure(
    go.Pie(
        labels=["Cubierto", "Faltante"],
        values=[covered, faltante],
        hole=0.55,  # aro más grueso
        textinfo="percent",
        textposition="inside",
        insidetextorientation="horizontal",
        sort=False,
        hovertemplate="%{label}: $%{value:,.0f}<br>%{percent}<extra></extra>",
        showlegend=False,
        marker=dict(
            colors=slice_colors,
            line=dict(color="rgba(255,255,255,0.5)", width=1)
        ),
        textfont=dict(color=text_colors),  # auto-contraste
    )
)
fig_donut.update_layout(
    title=f"{sem_icon} Cobertura de gastos {año_sel} · {cobertura_pct:.1f}%",
    margin=dict(t=60, l=20, r=20, b=20),
    annotations=[
        dict(
            text=str(año_sel), x=0.5, y=0.5, showarrow=False,
            font=dict(size=40, color=center_col)
        )
    ],
)
st.plotly_chart(fig_donut, use_container_width=True)


# ==========================
# Galería de donuts por año (auto size)
# ==========================
if ver_galeria:
    years_all = sorted(int(y) for y in resumen_all["Año"].unique() if 2025 <= int(y) <= 2030)
    if years_all:
        from math import ceil
        n = len(years_all)
        cols = min(cols_galeria or 4, n)
        rows = ceil(n / cols)
        specs = [[{"type": "domain"} for _ in range(cols)] for _ in range(rows)]

        fig_grid = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=[str(y) for y in years_all])
        for idx, y in enumerate(years_all, start=1):
            r = (idx - 1) // cols + 1
            c = (idx - 1) % cols + 1
            dfy = resumen_all[resumen_all["Año"] == y]
            cov_y = float(dfy["Total Ingresos"].sum() + dfy["Rentabilidad Inversiones"].sum())
            obj_y = float(dfy["Objetivo"].sum())
            covered_y = min(cov_y, obj_y)
            falt_y = max(obj_y - covered_y, 0.0)
            pct_y = 0.0 if obj_y <= 0 else (cov_y / obj_y) * 100.0

            fig_grid.add_trace(
                go.Pie(
                    labels=["Cubierto", "Faltante"],
                    values=[covered_y, falt_y],
                    hole=0.67,
                    textinfo="percent",
                    textposition="inside",
                    showlegend=False,
                    hovertemplate="%{label}: $%{value:,.0f}<br>%{percent}<extra></extra>",
                ),
                row=r, col=c,
            )
            fig_grid.add_annotation(
                x=(c - 0.5) / cols, y=1 - (r - 0.5) / rows, xref="paper", yref="paper",
                text=f"{pct_y:.1f}%", showarrow=False, font=dict(size=16, color=center_col)
            )

        fig_grid.update_layout(title="📆 Cobertura por año (2025–2030)", margin=dict(t=60, l=20, r=20, b=20))
        st.plotly_chart(fig_grid, use_container_width=True)
    else:
        st.info("No hay años 2025–2030 en el Excel para mostrar la galería.")

# ==========================
# Ingresos vs Gastos + línea de objetivo mensual (prom.)
# ==========================
fig_bar = go.Figure()
fig_bar.add_bar(name="Total Ingresos", x=resumen_y["Mes"], y=resumen_y["Total Ingresos"])
fig_bar.add_bar(name="Total Gastos", x=resumen_y["Mes"], y=resumen_y["Total Gastos"])
fig_bar.add_trace(
    go.Scatter(
        name="Objetivo mensual (prom.)",
        x=resumen_y["Mes"],
        y=[avg_obj_mensual] * len(resumen_y),
        mode="lines",
        line=dict(width=2, dash="dash"),
        hovertemplate="Objetivo prom.: $%{y:,.0f}<extra></extra>",
    )
)
fig_bar.update_layout(barmode="group", title="Ingresos vs Gastos (mensual) + Objetivo", margin=dict(t=60, l=10, r=10, b=20))
st.plotly_chart(fig_bar, use_container_width=True)

# ==========================
# Neto mensual (semáforo por mes: verde/rojo)
# ==========================
colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in resumen_y["Neto"]]
fig_net = go.Figure(go.Bar(x=resumen_y["Mes"], y=resumen_y["Neto"], marker_color=colors, hovertemplate="Neto: $%{y:,.0f}<extra></extra>", name="Neto mensual"))
fig_net.update_layout(title="🟢/🔴 Neto mensual (Ingresos + Rent. Inv. − Gastos)", margin=dict(t=60, l=10, r=10, b=20))
st.plotly_chart(fig_net, use_container_width=True)

# ==========================
# Rentabilidad inversiones vs total (marcas rojas si Neto < 0)  **FIX colores**
# ==========================
fig_line_rent = go.Figure()

# Series base
fig_line_rent.add_trace(
    go.Scatter(
        x=resumen_y["Mes"],
        y=resumen_y["Rentabilidad Inversiones"],
        mode="lines+markers",
        name="Rentabilidad Inversiones",
        hovertemplate="Mes: %{x}<br>Rent. Inv.: $%{y:,.0f}<extra></extra>",
    )
)
fig_line_rent.add_trace(
    go.Scatter(
        x=resumen_y["Mes"],
        y=resumen_y["Rentabilidad Total"],
        mode="lines+markers",
        name="Rentabilidad Total",
        hovertemplate="Mes: %{x}<br>Rent. Total: $%{y:,.0f}<extra></extra>",
    )
)

# Marcadores rojos SOLO donde Neto < 0 (sin None)
neg_mask = (resumen_y["Neto"] < 0).values
if neg_mask.any():
    fig_line_rent.add_trace(
        go.Scatter(
            x=resumen_y["Mes"][neg_mask],
            y=resumen_y["Rentabilidad Total"][neg_mask],
            mode="markers",
            name="Neto < 0",
            marker=dict(size=10, color="#E74C3C"),
            hovertemplate="Mes: %{x}<br>Rent. Total: $%{y:,.0f}<br>Neto: negativo<extra></extra>",
        )
    )

fig_line_rent.update_layout(
    title="Rentabilidad (mensual): Inversiones vs Total (marcas rojas = Neto < 0)",
    margin=dict(t=60, l=10, r=10, b=20),
)
st.plotly_chart(fig_line_rent, use_container_width=True)

# ==========================
# Rentabilidad acumulada (YTD)
# ==========================
fig_line_acum = px.line(resumen_y, x="Mes", y="Rentabilidad Acumulada", markers=True, title="Rentabilidad Acumulada (YTD)")
fig_line_acum.update_layout(margin=dict(t=60, l=10, r=10, b=20))
st.plotly_chart(fig_line_acum, use_container_width=True)

# ==========================
# Proyección probabilística (Monte Carlo)
# ==========================
st.subheader("🔮 Proyección probabilística de **cobertura** (3, 6, 12 meses)")

H_LIST = [3, 6, 12]
m_ini_idx = MESES_IDX[mes_inicio_mc]

pool_net = muestra_neta_para_mes_inicio(resumen_all, m_ini_idx, max(H_LIST))
if pool_net.size == 0:
    pool_net = np.array([0.0], dtype=float)

rng = np.random.default_rng(semilla)
mu = float(np.mean(pool_net)) if pool_net.size else 0.0
sigma = float(np.std(pool_net, ddof=1)) if pool_net.size > 1 else 0.0

@st.cache_data(show_spinner=False)
def simular_probabilidades(pool_tuple, horizontes, n_sims, metodo, seed):
    pool = np.array(pool_tuple, dtype=float)
    rng_local = np.random.default_rng(seed)
    resultados = {}
    for H in horizontes:
        if metodo == "Bootstrap (empírico)":
            draws = rng_local.choice(pool, size=(n_sims, H), replace=True) if pool.size else np.zeros((n_sims, H))
        else:
            m = float(np.mean(pool)) if pool.size else 0.0
            s = float(np.std(pool, ddof=1)) if pool.size > 1 else 0.0
            draws = rng_local.normal(m, s, size=(n_sims, H))
        cum_net = draws.sum(axis=1)
        prob_cover = float(np.mean(cum_net >= 0))
        resultados[H] = {"prob": prob_cover, "cum_net": cum_net}
    return resultados

resultados = simular_probabilidades(tuple(pool_net.tolist()), H_LIST, num_sims, metodo_mc, semilla)

def cobertura_pct_escenario(net_pm, obj_list):
    avg_obj = float(np.mean(obj_list)) if len(obj_list) else 0.0
    if avg_obj <= 0:
        return 100.0
    return 100.0 * (1.0 + (net_pm / avg_obj))

rows = []
for H in H_LIST:
    obj_list = objetivos_para_horizonte(resumen_all, año_sel, m_ini_idx, H)
    pess = cobertura_pct_escenario(mu - sigma, obj_list)
    base = cobertura_pct_escenario(mu, obj_list)
    opt = cobertura_pct_escenario(mu + sigma, obj_list)
    prob = resultados[H]["prob"] * 100
    rows.append({
        "Horizonte": f"{H} meses desde {mes_inicio_mc}",
        "Prob. cobertura ≥100%": f"{prob:.1f}%",
        "Pesimista (μ-σ)": f"{pess:.1f}%",
        "Base (μ)": f"{base:.1f}%",
        "Optimista (μ+σ)": f"{opt:.1f}%",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# Campana de Gauss (12 meses)
sim12 = resultados[12]["cum_net"] if 12 in resultados else np.array([0.0])
mean12 = float(np.mean(sim12)) if len(sim12) else 0.0
std12 = float(np.std(sim12, ddof=1)) if len(sim12) > 1 else 0.0

hist = go.Histogram(x=sim12, histnorm="probability density", nbinsx=40, name="Simulado")
if std12 > 0:
    xs = np.linspace(mean12 - 4 * std12, mean12 + 4 * std12, 200)
    pdf = (1.0 / (std12 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mean12) / std12) ** 2)
else:
    xs = np.array([mean12 - 1, mean12 + 1]); pdf = np.zeros_like(xs)

line = go.Scatter(x=xs, y=pdf, mode="lines", name="Normal(μ,σ)")
fig_gauss = go.Figure(data=[hist, line])
fig_gauss.update_layout(title=f"🔔 Neto acumulado en 12 meses (desde {mes_inicio_mc})", xaxis_title="Neto acumulado (12m)", yaxis_title="Densidad", margin=dict(t=60, l=20, r=20, b=20))
st.plotly_chart(fig_gauss, use_container_width=True)

# Conclusión ejecutiva
obj_list_12 = objetivos_para_horizonte(resumen_all, año_sel, m_ini_idx, 12)
base_cov_12 = cobertura_pct_escenario(mu, obj_list_12)
st.success(
    f"Conclusión: desde **{mes_inicio_mc}**, la probabilidad de **cubrir gastos** a 12 meses es **{(resultados[12]['prob']*100 if 12 in resultados else 0):.1f}%**. "
    f"Escenario base sugiere cobertura mensual esperada **{semaforo_color(base_cov_12, umbral_amarillo)} {base_cov_12:.1f}%**."
)

# ==========================
# Tabla mensual (opcional)
# ==========================
if mostrar_tabla:
    st.subheader("📋 Resumen mensual (detallado)")
    cols = [
        "Mes", "Año",
        "Total Ingresos", "Total Gastos",
        "Rentabilidad Inversiones", "Rentabilidad Total",
        "Rentabilidad Acumulada", "Objetivo",
    ]
    tabla = resumen_y[cols].copy()
    for c in ["Total Ingresos", "Total Gastos", "Rentabilidad Inversiones", "Rentabilidad Total", "Rentabilidad Acumulada", "Objetivo"]:
        tabla[c] = tabla[c].apply(moneda)
    st.dataframe(tabla, use_container_width=True)

# ==========================
# Notas para el usuario
# ==========================
st.caption(
    """Estructura requerida de hojas y columnas:
• Inversiones: Mes, Año, Activo, Monto Invertido, Rentabilidad (%), Valor Actual, Rentabilidad Real.
• Ingresos Profesionales: Mes, Año, Concepto, Monto.
• Gastos Fijos: Mes, Año, Concepto, Monto.
• Resumen Mensual: Mes, Año, Total Ingresos, Total Gastos, Rentabilidad Inversiones, Rentabilidad Total, Rentabilidad Acumulada, Objetivo."""
)
