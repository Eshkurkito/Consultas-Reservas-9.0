import io
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =====================================
# Utilidades comunes
# =====================================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Precio"]
    for col in required:
        if col not in df.columns:
            st.error(f"Falta la columna obligatoria: {col}")
            st.stop()
    df["Fecha alta"] = pd.to_datetime(df["Fecha alta"], errors="coerce")
    df["Fecha entrada"] = pd.to_datetime(df["Fecha entrada"], errors="coerce")
    df["Fecha salida"] = pd.to_datetime(df["Fecha salida"], errors="coerce")
    df["Alojamiento"] = df["Alojamiento"].astype(str).str.strip()
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def load_excel_from_blobs(file_blobs: List[tuple[str, bytes]]) -> pd.DataFrame:
    """Carga y concatena varios Excel a partir de blobs (nombre, bytes). Se cachea por contenido."""
    frames = []
    for name, data in file_blobs:
        try:
            xls = pd.ExcelFile(io.BytesIO(data))
            sheet = (
                "Estado de pagos de las reservas"
                if "Estado de pagos de las reservas" in xls.sheet_names
                else xls.sheet_names[0]
            )
            df = pd.read_excel(xls, sheet_name=sheet)
            df["__source_file__"] = name
            frames.append(df)
        except Exception as e:
            st.error(f"No se pudo leer {name}: {e}")
            st.stop()
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    return parse_dates(df_all)

# --- Cálculo vectorizado de KPIs (rápido)

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Calcula KPIs de forma vectorizada sin expandir noche a noche."""
    # 1) Filtrar por corte y propiedades
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]

    # Quitar filas sin fechas válidas
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    if df_cut.empty:
        inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
        if inventory_override and inventory_override > 0:
            inv = int(inventory_override)
        days = (period_end - period_start).days + 1
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": inv * days,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))  # fin inclusivo

    arr_e = df_cut["Fecha entrada"].values.astype('datetime64[ns]')
    arr_s = df_cut["Fecha salida"].values.astype('datetime64[ns]')
    arr_c = df_cut["Fecha alta"].values.astype('datetime64[ns]')

    total_nights = ((arr_s - arr_e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype('int64')
    ov_days = np.clip(ov_days, 0, None)

    if ov_days.sum() == 0:
        inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
        if inventory_override and inventory_override > 0:
            inv = int(inventory_override)
        days = (period_end - period_start).days + 1
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": inv * days,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    price = df_cut["Precio"].values.astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total_nights > 0, ov_days / total_nights, 0.0)
    income = price * share

    props = df_cut["Alojamiento"].astype(str).values
    df_agg = pd.DataFrame({"Alojamiento": props, "Noches": ov_days, "Ingresos": income})
    by_prop = df_agg.groupby("Alojamiento", as_index=False).sum(numeric_only=True)
    by_prop.rename(columns={"Noches": "Noches ocupadas"}, inplace=True)
    by_prop["ADR"] = np.where(by_prop["Noches ocupadas"] > 0, by_prop["Ingresos"] / by_prop["Noches ocupadas"], 0.0)
    by_prop = by_prop.sort_values("Alojamiento")

    noches_ocupadas = int(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = float(ingresos / noches_ocupadas) if noches_ocupadas > 0 else 0.0

    inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    if inventory_override and inventory_override > 0:
        inv = int(inventory_override)
    days = (period_end - period_start).days + 1
    noches_disponibles = inv * days
    ocupacion_pct = (noches_ocupadas / noches_disponibles * 100) if noches_disponibles > 0 else 0.0
    revpar = ingresos / noches_disponibles if noches_disponibles > 0 else 0.0

    tot = {
        "noches_ocupadas": noches_ocupadas,
        "noches_disponibles": noches_disponibles,
        "ocupacion_pct": ocupacion_pct,
        "ingresos": ingresos,
        "adr": adr,
        "revpar": revpar,
    }

    return by_prop, tot

# -------------------------------------
# Helpers adicionales
# -------------------------------------

def help_block(kind: str):
    """Bloque de ayuda contextual por sección."""
    texts = {
        "Consulta normal": """
**Qué es:** KPIs del periodo elegido **a la fecha de corte**.
- *Noches ocupadas*: noches del periodo que caen dentro de reservas con **Fecha alta ≤ corte**.
- *Noches disponibles*: inventario × nº de días del periodo (puedes **sobrescribir inventario**).
- *Ocupación %* = Noches ocupadas / Noches disponibles.
- *Ingresos* = precio prorrateado por noche dentro del periodo.
- *ADR* = Ingresos / Noches ocupadas.
- *RevPAR* = Ingresos / Noches disponibles.
**Tips:** Filtra alojamientos o compara con año anterior (mismo periodo y corte-1 año) para una lectura YoY.
        """,
        "KPIs por meses": """
**Qué es:** Serie por **meses seleccionados** con KPIs a la **misma fecha de corte**.
**Cómo leer:** cada punto/mes muestra el KPI consolidado del mes. Con YoY activado, verás otra línea desplazada -1 año.
**Uso típico:** ver estacionalidad de *Ocupación*, *ADR* o *RevPAR* a un corte único y detectar meses retrasados/adelantados.
        """,
        "Evolución por corte": """
**Qué es:** Cómo **crecen** los KPIs del mismo periodo cuando **mueves la fecha de corte** día a día.
**Cómo leer:** si la línea sube rápido → buen *pickup*; si se aplana → ritmo débil. Con YoY verás la evolución del año pasado alineada por día.
        """,
        "Pickup": """
**Qué es:** Diferencia entre dos cortes A y B (**B – A**).
- Vista **Diario**: incremento día a día.
- Vista **Acumulado**: total ganado desde A.
**KPIs:** Δ Noches, Δ Ingresos, Δ Ocupación, Δ ADR, Δ RevPAR. Tabla *Top alojamientos* por pickup.
**Interpretación:** Δ Noches positivo con ADR estable/subiendo = acción de precio acertada; Δ Noches negativo y ADR↑ = sobreprecio.
        """,
        "Pace": """
**Qué es:** KPI confirmado a **D días antes de la estancia** (D=0 día de llegada).
**Cómo leer:** compara tu curva con LY. Por ejemplo, a D-60 deberías tener X% de tu objetivo si sigues patrón normal.
**Uso:** detectar retrasos/adelantos y ajustar precio y mínimos.
        """,
        "Predicción": """
**Qué es:** Forecast por Pace con banda **[P25–P75]** de noches finales.
**Tarjetas:** OTB, *Forecast P50* (noches/ingresos), *ADR final (P50)*, **Pickup necesario** vs **Pickup típico** (semáforo).
**Cómo actuar:**
- 🟢 Necesario ≤ típico P50 → mantener/subir.
- 🟠 Entre P50 y P75 → cautela.
- 🔴 > P75 → ajustar precio/oferta.
        """,
        "Lead": """
**Lead time:** días entre **Fecha alta** y **Fecha entrada** (por reserva). Muestra media, mediana y distribución.
**LOS:** noches por reserva. Útil para reglas de mínimos, ofertas y limpieza.
        """,
        "DOW": """
**Qué es:** Calor por **Día de la semana × Mes**.
- *Ocupación (noches)*: cuántas noches vendidas tiene cada DOW por mes.
- *Ocupación (%)*: noches / (inventario × nº de ese DOW en el mes).
- *ADR (€)*: precio medio de esas noches.
**Uso:** fija mínimos y precios por DOW; detecta días sistemáticamente caros/baratos.
        """,
        "ADR bands": """
**Qué es:** percentiles **P10/P25/Mediana/P75/P90** del ADR por **reserva** para cada mes.
**Extra:** línea **ADR OTB** y tabla con **posición Pxx** del mes.
**Uso:** define barandillas de precio: base en P50–P65, picos cerca de P75–P90, suelo ≈ P25.
        """,
        "Calendario": """
**Qué es:** matriz Alojamiento × Día.
- *Ocupado/Libre*: bloque ■ si hay noche ocupada.
- *ADR*: ADR por noche (prorrateado) en cada celda.
**Uso:** detectar huecos, solapes y variaciones de precio a nivel micro.
        """,
        "Estacionalidad": """
**Qué es:** distribución por **Mes del año**, **Día de la semana** o **Día del mes**.
**Base:** por *Noches (estancia)* o *Reservas (check-in)*. Opción de **índice** (1=media) para ver forma sin volumen.
**Uso:** planning de campañas y expectativas por patrón.
        """,
    }
    txt = texts.get(kind, None)
    if txt:
        with st.expander("ℹ️ Cómo leer esta sección", expanded=False):
            st.markdown(txt)


def period_inputs(label_start: str, label_end: str, default_start: date, default_end: date, key_prefix: str) -> tuple[date, date]:
    """Par de date_input sincronizado con el periodo global si el interruptor 'keep_period' está activo.
    Devuelve (start, end) como objetos date.
    """
    keep = st.session_state.get("keep_period", False)
    g_start = st.session_state.get("global_period_start")
    g_end = st.session_state.get("global_period_end")
    val_start = g_start if (keep and g_start) else default_start
    val_end = g_end if (keep and g_end) else default_end
    c1, c2 = st.columns(2)
    with c1:
        start_val = st.date_input(label_start, value=val_start, key=f"{key_prefix}_start")
    with c2:
        end_val = st.date_input(label_end, value=val_end, key=f"{key_prefix}_end")
    if keep:
        st.session_state["global_period_start"] = start_val
        st.session_state["global_period_end"] = end_val
    return start_val, end_val

def period_inputs(label_start: str, label_end: str, default_start: date, default_end: date, key_prefix: str) -> tuple[date, date]:
    """Par de date_input sincronizado con el periodo global si el interruptor 'keep_period' está activo.
    Devuelve (start, end) como objetos date.
    """
    keep = st.session_state.get("keep_period", False)
    g_start = st.session_state.get("global_period_start")
    g_end = st.session_state.get("global_period_end")
    val_start = g_start if (keep and g_start) else default_start
    val_end = g_end if (keep and g_end) else default_end
    c1, c2 = st.columns(2)
    with c1:
        start_val = st.date_input(label_start, value=val_start, key=f"{key_prefix}_start")
    with c2:
        end_val = st.date_input(label_end, value=val_end, key=f"{key_prefix}_end")
    if keep:
        st.session_state["global_period_start"] = start_val
        st.session_state["global_period_end"] = end_val
    return start_val, end_val

def get_inventory(df: pd.DataFrame, override: Optional[int]) -> int:
    inv = df["Alojamiento"].nunique()
    if override and override > 0:
        inv = int(override)
    return int(inv)

def occurrences_of_dow_by_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start, end, freq='D')
    df = pd.DataFrame({"Fecha": days})
    df["Mes"] = df["Fecha"].dt.to_period('M').astype(str)
    df["DOW"] = df["Fecha"].dt.weekday.map({0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"})
    occ = df.groupby(["DOW","Mes"]).size().reset_index(name="occ")
    return occ

def pace_profiles_for_refs(df: pd.DataFrame, target_start: pd.Timestamp, target_end: pd.Timestamp, ref_years: int, dmax: int, props: Optional[List[str]] = None, inv_override: Optional[int] = None) -> dict:
    """Devuelve perfiles F(D) de noches para los meses de referencia (mismo mes en años anteriores).
    Retorna dict con keys: 'F25','F50','F75' (arrays de tamaño dmax+1).
    """
    profiles = []
    for k in range(1, ref_years+1):
        s = target_start - pd.DateOffset(years=k)
        e = target_end - pd.DateOffset(years=k)
        base = pace_series(df, s, e, dmax, props, inv_override)
        if base.empty or base['noches'].max() == 0:
            continue
        final_n = base.loc[base['D']==0, 'noches'].values[0]
        if final_n <= 0:
            continue
        F = base['noches'] / final_n
        profiles.append(F.values)
    if not profiles:
        # fallback: perfil lineal suave
        F = np.linspace(0.2, 1.0, dmax+1)
        return {"F25": F, "F50": F, "F75": F}
    M = np.vstack(profiles)
    F25 = np.nanpercentile(M, 25, axis=0)
    F50 = np.nanpercentile(M, 50, axis=0)
    F75 = np.nanpercentile(M, 75, axis=0)
    return {"F25": F25, "F50": F50, "F75": F75}

def pace_forecast_month(df: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, ref_years: int = 2, dmax: int = 180, props: Optional[List[str]] = None, inv_override: Optional[int] = None) -> dict:
    """Forecast mensual por Pace usando perfiles F(D) históricos y OTB diario.
    Devuelve dict con: nights_otb, nights_p25, nights_p50, nights_p75,
    adr_final_p50, revenue_final_p50, adr_tail_p25/p50/p75,
    pickup_needed_p50, pickup_typ_p50/p75 (escalados al tamaño actual), daily, n_refs.
    """
    # OTB diario a la fecha de corte
    daily = daily_series(df, pd.to_datetime(cutoff), start, end, props, inv_override)
    daily = daily.sort_values('Fecha')

    # D para cada día (lead hasta la noche de estancia)
    D_day = (daily['Fecha'] - pd.to_datetime(cutoff)).dt.days.clip(lower=0)
    dmax = int(max(dmax, D_day.max()))

    # Perfiles F(D) históricos (P25/P50/P75)
    prof = pace_profiles_for_refs(df, start, end, ref_years, dmax, props, inv_override)
    F25, F50, F75 = prof['F25'], prof['F50'], prof['F75']

    # Helper para obtener F en el D correspondiente de cada día
    def f_at(arr, d):
        d = int(min(max(d, 0), len(arr)-1))
        return float(arr[d]) if not np.isnan(arr[d]) else 1.0

    eps = 1e-6
    daily['D'] = D_day
    daily['F25'] = daily['D'].apply(lambda d: f_at(F25, d))
    daily['F50'] = daily['D'].apply(lambda d: f_at(F50, d))
    daily['F75'] = daily['D'].apply(lambda d: f_at(F75, d))
    daily['n_final_p25'] = daily['noches_ocupadas'] / daily['F25'].clip(lower=eps)
    daily['n_final_p50'] = daily['noches_ocupadas'] / daily['F50'].clip(lower=eps)
    daily['n_final_p75'] = daily['noches_ocupadas'] / daily['F75'].clip(lower=eps)

    nights_otb = float(daily['noches_ocupadas'].sum())
    nights_p25 = float(daily['n_final_p25'].sum())
    nights_p50 = float(daily['n_final_p50'].sum())
    nights_p75 = float(daily['n_final_p75'].sum())

    # ADR OTB y revenue OTB actuales
    _, tot_now = compute_kpis(df, pd.to_datetime(cutoff), start, end, inv_override, props)
    adr_otb = float(tot_now['adr'])
    rev_otb = float(tot_now['ingresos'])

    # Estimar ADR + NIGHTS del remanente a partir de años de referencia
    D_med = int(np.median(D_day))
    tail_adrs, tail_nights, finals_hist = [], [], []
    n_refs_used = 0
    for k in range(1, ref_years+1):
        s = start - pd.DateOffset(years=k)
        e = end - pd.DateOffset(years=k)
        base = pace_series(df, s, e, max(D_med, 0), props, inv_override)
        if base.empty:
            continue
        n_refs_used += 1
        if 0 not in base['D'].values:
            continue
        nights_final = float(base.loc[base['D']==0, 'noches'].values[0])
        rev_final = float(base.loc[base['D']==0, 'ingresos'].values[0])
        finals_hist.append(nights_final)
        if D_med in base['D'].values:
            nights_atD = float(base.loc[base['D']==D_med, 'noches'].values[0])
            rev_atD = float(base.loc[base['D']==D_med, 'ingresos'].values[0])
        else:
            nights_atD = float('nan'); rev_atD = float('nan')
        dn = max(nights_final - (nights_atD if np.isfinite(nights_atD) else 0.0), 0.0)
        dr = max(rev_final - (rev_atD if np.isfinite(rev_atD) else 0.0), 0.0)
        if dn > 0:
            tail_adrs.append(dr/dn)
            tail_nights.append(dn)

    if tail_adrs:
        adr_tail_p25 = float(np.percentile(tail_adrs, 25))
        adr_tail_p50 = float(np.percentile(tail_adrs, 50))
        adr_tail_p75 = float(np.percentile(tail_adrs, 75))
    else:
        adr_tail_p25 = adr_tail_p50 = adr_tail_p75 = adr_otb

    # Escalar pickup típico de noches a tamaño actual del mes
    if tail_nights and finals_hist and np.median(finals_hist) > 0:
        scale = nights_p50 / float(np.median(finals_hist))
        pickup_typ_p50 = float(np.percentile(tail_nights, 50)) * scale
        pickup_typ_p75 = float(np.percentile(tail_nights, 75)) * scale
    else:
        # fallback al rango de banda P75..P25
        pickup_typ_p50 = max(nights_p50 - nights_otb, 0.0)
        pickup_typ_p75 = max(nights_p25 - nights_otb, 0.0)

    # Forecast P50 de ingresos/ADR final
    nights_rem_p50 = max(nights_p50 - nights_otb, 0.0)
    revenue_final_p50 = rev_otb + adr_tail_p50 * nights_rem_p50
    adr_final_p50 = revenue_final_p50 / nights_p50 if nights_p50 > 0 else 0.0

    # Pickup necesario hacia P50
    pickup_needed_p50 = nights_rem_p50

    return {
        "nights_otb": nights_otb,
        "nights_p25": nights_p25,
        "nights_p50": nights_p50,
        "nights_p75": nights_p75,
        "adr_final_p50": adr_final_p50,
        "revenue_final_p50": revenue_final_p50,
        "adr_tail_p25": adr_tail_p25,
        "adr_tail_p50": adr_tail_p50,
        "adr_tail_p75": adr_tail_p75,
        "pickup_needed_p50": pickup_needed_p50,
        "pickup_typ_p50": pickup_typ_p50,
        "pickup_typ_p75": pickup_typ_p75,
        "daily": daily,
        "n_refs": n_refs_used,
    }

def compute_portal_share(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    filter_props: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Distribución por portal (porcentaje de reservas que intersectan el periodo a la fecha de corte).
    Devuelve DataFrame con columnas: Portal, Reservas, % Reservas. Si no hay columna 'Portal', retorna None."""
    if "Portal" not in df_all.columns:
        return None

    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df = df[df["Alojamiento"].isin(filter_props)]
    df = df.dropna(subset=["Fecha entrada", "Fecha salida", "Portal"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Portal", "Reservas", "% Reservas"]) 

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))

    arr_e = df["Fecha entrada"].values.astype('datetime64[ns]')
    arr_s = df["Fecha salida"].values.astype('datetime64[ns]')

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype('int64')
    mask = ov_days > 0
    if mask.sum() == 0:
        return pd.DataFrame(columns=["Portal", "Reservas", "% Reservas"]) 

    df_sel = df.loc[mask]
    counts = df_sel.groupby("Portal").size().reset_index(name="Reservas").sort_values("Reservas", ascending=False)
    total = counts["Reservas"].sum()
    counts["% Reservas"] = np.where(total > 0, counts["Reservas"] / total * 100.0, 0.0)
    return counts

def daily_series(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]], inventory_override: Optional[int]) -> pd.DataFrame:
    """Devuelve serie diaria de KPIs (noches, ingresos, ocupación %, ADR, RevPAR)."""
    days = list(pd.date_range(start, end, freq='D'))
    rows = []
    for d in days:
        _bp, tot = compute_kpis(
            df_all=df_all,
            cutoff=cutoff,
            period_start=d,
            period_end=d,
            inventory_override=inventory_override,
            filter_props=props,
        )
        rows.append({"Fecha": d.normalize(), **tot})
    return pd.DataFrame(rows)


def build_calendar_matrix(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]], mode: str = "Ocupado/Libre") -> pd.DataFrame:
    """Matriz (alojamientos x días) con '■' si ocupado o ADR por noche si mode='ADR'."""
    df_cut = df_all[(df_all["Fecha alta"] <= cutoff)].copy()
    if props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"])
    if df_cut.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_cut.iterrows():
        e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
        ov_start = max(e, start)
        ov_end = min(s, end + pd.Timedelta(days=1))
        n_nights = (s - e).days
        if ov_start >= ov_end or n_nights <= 0:
            continue
        adr_night = p / n_nights if n_nights > 0 else 0.0
        for d in pd.date_range(ov_start, ov_end - pd.Timedelta(days=1), freq='D'):
            rows.append({"Alojamiento": r["Alojamiento"], "Fecha": d.normalize(), "Ocupado": 1, "ADR_noche": adr_night})
    if not rows:
        return pd.DataFrame()
    df_nightly = pd.DataFrame(rows)

    if mode == "Ocupado/Libre":
        piv = df_nightly.pivot_table(index="Alojamiento", columns="Fecha", values="Ocupado", aggfunc='sum', fill_value=0)
        piv = piv.applymap(lambda x: '■' if x > 0 else '')
    else:
        piv = df_nightly.pivot_table(index="Alojamiento", columns="Fecha", values="ADR_noche", aggfunc='mean', fill_value='')

    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv


def pace_series(df_all: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp, d_max: int, props: Optional[List[str]], inv_override: Optional[int]) -> pd.DataFrame:
    """Curva Pace: para cada D (0..d_max), noches/ingresos confirmados a D días antes de la estancia.
    Fórmula por reserva: noches(D) = len( [max(ov_start, created_at + D), ov_end) ) en días
    donde ov_* es la intersección de [entrada, salida) con [period_start, period_end+1).
    """
    df = df_all.dropna(subset=["Fecha alta", "Fecha entrada", "Fecha salida"]).copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    if df.empty:
        return pd.DataFrame({"D": list(range(d_max + 1)), "noches": 0, "ingresos": 0.0, "ocupacion_pct": 0.0, "adr": 0.0, "revpar": 0.0})

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))

    e = df["Fecha entrada"].values.astype('datetime64[ns]')
    s = df["Fecha salida"].values.astype('datetime64[ns]')
    c = df["Fecha alta"].values.astype('datetime64[ns]')
    price = df["Precio"].values.astype('float64')

    total_nights = ((s - e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)
    adr_night = np.where(total_nights > 0, price / total_nights, 0.0)

    ov_start = np.maximum(e, start_ns)
    ov_end = np.minimum(s, end_excl_ns)
    valid = (ov_end > ov_start) & (total_nights > 0)
    if not valid.any():
        inv = len(set(props)) if props else df_all["Alojamiento"].nunique()
        if inv_override and inv_override > 0:
            inv = int(inv_override)
        days = (period_end - period_start).days + 1
        return pd.DataFrame({"D": list(range(d_max + 1)), "noches": 0, "ingresos": 0.0, "ocupacion_pct": 0.0, "adr": 0.0, "revpar": 0.0})

    e = e[valid]; s = s[valid]; c = c[valid]; ov_start = ov_start[valid]; ov_end = ov_end[valid]; adr_night = adr_night[valid]

    D_vals = np.arange(0, d_max + 1, dtype='int64')
    D_td = D_vals * one_day  # shape (D,)

    # start threshold por D: c + D
    start_thr = c[:, None] + D_td[None, :]
    ov_start_b = np.maximum(ov_start[:, None], start_thr)  # (n, D)
    nights_D = ((ov_end[:, None] - ov_start_b) / one_day).astype('int64')
    nights_D = np.clip(nights_D, 0, None)

    nights_series = nights_D.sum(axis=0).astype(float)
    ingresos_series = (nights_D * adr_night[:, None]).sum(axis=0)

    inv = len(set(props)) if props else df_all["Alojamiento"].nunique()
    if inv_override and inv_override > 0:
        inv = int(inv_override)
    days = (period_end - period_start).days + 1
    disponibles = inv * days if days > 0 else 0

    occ_series = (nights_series / disponibles * 100.0) if disponibles > 0 else np.zeros_like(nights_series)
    adr_series = np.where(nights_series > 0, ingresos_series / nights_series, 0.0)
    revpar_series = (ingresos_series / disponibles) if disponibles > 0 else np.zeros_like(ingresos_series)

    return pd.DataFrame({
        "D": D_vals,
        "noches": nights_series,
        "ingresos": ingresos_series,
        "ocupacion_pct": occ_series,
        "adr": adr_series,
        "revpar": revpar_series,
    })

# =====================================
# App (con archivos persistentes en sesión)
# =====================================

st.set_page_config(page_title="Consultas OTB por corte", layout="wide")
st.title("📅 Consultas OTB – Ocupación, ADR y RevPAR a fecha de corte")
st.caption("Sube archivos una vez y úsalos en cualquiera de los modos.")

# --- Gestor de archivos global ---
with st.sidebar:
    # 🔁 Periodo global opcional
    st.checkbox("🧲 Mantener periodo entre modos", value=st.session_state.get("keep_period", False), key="keep_period", help="Si está activo, el periodo (inicio/fin) se guarda y se reutiliza en todos los modos.")
    colp1, colp2 = st.columns(2)
    with colp1:
        if st.button("Reset periodo"):
            st.session_state.pop("global_period_start", None)
            st.session_state.pop("global_period_end", None)
            st.success("Periodo global reiniciado")
    with colp2:
        if st.session_state.get("keep_period"):
            st.caption(f"Periodo actual: {st.session_state.get('global_period_start','–')} → {st.session_state.get('global_period_end','–')}")

    st.header("Archivos de trabajo (persisten en la sesión)")
    files_master = st.file_uploader(
        "Sube uno o varios Excel",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="files_master",
        help="Se admiten múltiples años (2024, 2025…).",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Usar estos archivos", type="primary"):
            if files_master:
                blobs = [(f.name, f.getvalue()) for f in files_master]
                df_loaded = load_excel_from_blobs(blobs)
                st.session_state["raw_df"] = df_loaded
                st.session_state["file_names"] = [n for n, _ in blobs]
                st.success(f"Cargados {len(blobs)} archivo(s)")
            else:
                st.warning("No seleccionaste archivos.")
    with col_b:
        if st.button("Limpiar archivos"):
            st.session_state.pop("raw_df", None)
            st.session_state.pop("file_names", None)
            st.info("Archivos eliminados de la sesión.")

# Targets opcionales (persisten en sesión)
with st.sidebar.expander("🎯 Cargar Targets (opcional)"):
    tgt_file = st.file_uploader("CSV Targets", type=["csv"], key="tgt_csv")
    if tgt_file is not None:
        try:
            df_tgt = pd.read_csv(tgt_file)
            # Normalizar columnas esperadas si existen
            # year, month, target_occ_pct, target_adr, target_revpar, target_nights, target_revenue
            st.session_state["targets_df"] = df_tgt
            st.success("Targets cargados en sesión.")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

raw = st.session_state.get("raw_df")
file_names = st.session_state.get("file_names", [])

if raw is not None:
    with st.expander("📂 Archivos cargados"):
        st.write("**Lista:**", file_names)
        st.write(f"**Alojamientos detectados:** {raw['Alojamiento'].nunique()}")
else:
    st.info("Sube archivos en la barra lateral y pulsa **Usar estos archivos** para empezar.")

# -----------------------------
# Menú de modos (independientes)
# -----------------------------
mode = st.sidebar.radio(
    "Modo de consulta",
    [
        "Consulta normal",
        "KPIs por meses",
        "Evolución por fecha de corte",
        "Pickup (entre dos cortes)",
        "Pace (curva D)",
        "Predicción (Pace)",
        "Pipeline 90–180 días",
        "Gap vs Target",
        "Lead time & LOS",
        "DOW heatmap",
        "ADR bands & Targets",
        "Pricing – Mapa eficiencia",
        "Cohortes (Alta × Estancia)",
        "Estacionalidad",
        "Ranking alojamientos",
        "Operativa",
        "Calidad de datos",
        "Calendario por alojamiento",
    ],
)

# Helper para mapear nombres de métricas a columnas
METRIC_MAP = {"Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}

# =============================
# MODO 1: Consulta normal (+ comparación año anterior con inventario propio)
# =============================
if mode == "Consulta normal":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_normal = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cutoff_normal")
        c1, c2 = st.columns(2)
        start_normal, end_normal = period_inputs("Inicio del periodo", "Fin del periodo", date(2024, 9, 1), date(2024, 9, 30), "normal")
        inv_normal = st.number_input("Sobrescribir inventario (nº alojamientos)", min_value=0, value=0, step=1, key="inv_normal")
        props_normal = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_normal")
        st.markdown("—")
        compare_normal = st.checkbox("Comparar con año anterior (mismo día/mes)", value=False, key="cmp_normal")
        inv_normal_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_normal_prev")

    # Cálculo base
    by_prop_n, total_n = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        inventory_override=int(inv_normal) if inv_normal > 0 else None,
        filter_props=props_normal if props_normal else None,
    )

    st.subheader("Resultados totales")
    help_block("Consulta normal")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_n['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_n['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_n['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{total_n['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{total_n['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{total_n['revpar']:.2f}")
    # --- Distribución por portal (reservas del periodo a la fecha de corte)
    port_df = compute_portal_share(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        filter_props=props_normal if props_normal else None,
    )
    st.subheader("Distribución por portal (reservas en el periodo)")
    if port_df is None:
        st.info("No se encontró la columna 'Portal' (o 'Canal') en los archivos. Si tiene otro nombre, dímelo y lo mapeo.")
    elif port_df.empty:
        st.warning("No hay reservas del periodo a la fecha de corte para calcular distribución por portal.")
    else:
        port_view = port_df.copy()
        port_view["% Reservas"] = port_view["% Reservas"].round(2)
        st.dataframe(port_view, use_container_width=True)
        csv_port = port_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar distribución por portal (CSV)", data=csv_port, file_name="portales_distribucion.csv", mime="text/csv")

    if compare_normal:
        cutoff_cmp = (pd.to_datetime(cutoff_normal) - pd.DateOffset(years=1)).date()
        start_cmp = (pd.to_datetime(start_normal) - pd.DateOffset(years=1)).date()
        end_cmp = (pd.to_datetime(end_normal) - pd.DateOffset(years=1)).date()
        _bp_c, total_cmp = compute_kpis(
            df_all=raw,
            cutoff=pd.to_datetime(cutoff_cmp),
            period_start=pd.to_datetime(start_cmp),
            period_end=pd.to_datetime(end_cmp),
            inventory_override=int(inv_normal_prev) if inv_normal_prev > 0 else None,
            filter_props=props_normal if props_normal else None,
        )
        st.markdown("**Comparativa con año anterior** (corte y periodo -1 año)")
        d1, d2, d3 = st.columns(3)
        d4, d5, d6 = st.columns(3)
        d1.metric("Noches ocupadas (prev)", f"{total_cmp['noches_ocupadas']:,}".replace(",", "."), delta=total_n['noches_ocupadas']-total_cmp['noches_ocupadas'])
        d2.metric("Noches disp. (prev)", f"{total_cmp['noches_disponibles']:,}".replace(",", "."), delta=total_n['noches_disponibles']-total_cmp['noches_disponibles'])
        d3.metric("Ocupación (prev)", f"{total_cmp['ocupacion_pct']:.2f}%", delta=f"{total_n['ocupacion_pct']-total_cmp['ocupacion_pct']:.2f}%")
        d4.metric("Ingresos (prev)", f"{total_cmp['ingresos']:.2f}", delta=f"{total_n['ingresos']-total_cmp['ingresos']:.2f}")
        d5.metric("ADR (prev)", f"{total_cmp['adr']:.2f}", delta=f"{total_n['adr']-total_cmp['adr']:.2f}")
        d6.metric("RevPAR (prev)", f"{total_cmp['revpar']:.2f}", delta=f"{total_n['revpar']-total_cmp['revpar']:.2f}")

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_n.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_n, use_container_width=True)
        csv = by_prop_n.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar detalle (CSV)", data=csv, file_name="detalle_por_alojamiento.csv", mime="text/csv")

# =============================
# MODO 2: KPIs por meses (línea) + comparación con inventario previo
# =============================
elif mode == "KPIs por meses":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_m = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cutoff_months")
        props_m = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_months")
        inv_m = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_months")
        inv_m_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_months_prev")
        _min = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).min()
        _max = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).max()
        months_options = [str(p) for p in pd.period_range(_min.to_period("M"), _max.to_period("M"), freq="M")] if pd.notna(_min) and pd.notna(_max) else []
        selected_months_m = st.multiselect("Meses a graficar (YYYY-MM)", options=months_options, default=[], key="months_months")
        metric_choice = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"])
        compare_m = st.checkbox("Comparar con año anterior (mismo mes)", value=False, key="cmp_months")

    st.subheader("📈 KPIs por meses (a fecha de corte)")
    help_block("KPIs por meses")
    if selected_months_m:
        rows_actual = []
        rows_prev = []
        for ym in selected_months_m:
            p = pd.Period(ym, freq="M")
            start_m = p.to_timestamp(how="start")
            end_m = p.to_timestamp(how="end")
            _bp, _tot = compute_kpis(
                df_all=raw,
                cutoff=pd.to_datetime(cutoff_m),
                period_start=start_m,
                period_end=end_m,
                inventory_override=int(inv_m) if inv_m > 0 else None,
                filter_props=props_m if props_m else None,
            )
            rows_actual.append({"Mes": ym, **_tot})

            if compare_m:
                p_prev = p - 12
                start_prev = p_prev.to_timestamp(how="start")
                end_prev = p_prev.to_timestamp(how="end")
                cutoff_prev = pd.to_datetime(cutoff_m) - pd.DateOffset(years=1)
                _bp2, _tot_prev = compute_kpis(
                    df_all=raw,
                    cutoff=cutoff_prev,
                    period_start=start_prev,
                    period_end=end_prev,
                    inventory_override=int(inv_m_prev) if inv_m_prev > 0 else None,
                    filter_props=props_m if props_m else None,
                )
                rows_prev.append({"Mes": ym, **_tot_prev})

        df_actual = pd.DataFrame(rows_actual).sort_values("Mes")
        if not compare_m:
            st.line_chart(df_actual.set_index("Mes")[ [ METRIC_MAP[metric_choice] ] ].rename(columns={METRIC_MAP[metric_choice]: metric_choice}), height=280)
            st.dataframe(df_actual[["Mes", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]].rename(columns={
                "noches_ocupadas": "Noches ocupadas",
                "noches_disponibles": "Noches disponibles",
                "ocupacion_pct": "Ocupación %",
                "adr": "ADR (€)",
                "revpar": "RevPAR (€)",
                "ingresos": "Ingresos (€)"
            }), use_container_width=True)
        else:
            df_prev = pd.DataFrame(rows_prev).sort_values("Mes") if rows_prev else pd.DataFrame()
            plot_df = pd.DataFrame({
                "Actual": df_actual[METRIC_MAP[metric_choice]].values
            }, index=df_actual["Mes"])
            if not df_prev.empty:
                plot_df["Año anterior"] = df_prev[METRIC_MAP[metric_choice]].values
            st.line_chart(plot_df, height=280)

            table_df = df_actual.merge(df_prev, on="Mes", how="left", suffixes=("", " (prev)")) if not df_prev.empty else df_actual
            rename_map = {
                "noches_ocupadas": "Noches ocupadas",
                "noches_disponibles": "Noches disponibles",
                "ocupacion_pct": "Ocupación %",
                "adr": "ADR (€)",
                "revpar": "RevPAR (€)",
                "ingresos": "Ingresos (€)",
                "noches_ocupadas (prev)": "Noches ocupadas (prev)",
                "noches_disponibles (prev)": "Noches disponibles (prev)",
                "ocupacion_pct (prev)": "Ocupación % (prev)",
                "adr (prev)": "ADR (€) (prev)",
                "revpar (prev)": "RevPAR (€) (prev)",
                "ingresos (prev)": "Ingresos (€) (prev)",
            }
            st.dataframe(table_df.rename(columns=rename_map), use_container_width=True)

        csvm = df_actual.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar KPIs por mes (CSV)", data=csvm, file_name="kpis_por_mes.csv", mime="text/csv")
    else:
        st.info("Selecciona meses en la barra lateral para ver la gráfica.")

# =============================
# MODO 3: Evolución por fecha de corte + comparación con inventario previo
# =============================
elif mode == "Evolución por fecha de corte":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Rango de corte")
        evo_cut_start = st.date_input("Inicio de corte", value=date(2024, 4, 1), key="evo_cut_start_new")
        evo_cut_end = st.date_input("Fin de corte", value=date(2024, 4, 30), key="evo_cut_end_new")

        st.header("Periodo objetivo")
        evo_target_start, evo_target_end = period_inputs("Inicio del periodo", "Fin del periodo", date(2024, 9, 1), date(2024, 9, 30), "evo_target")

        props_e = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_evo")
        inv_e = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_evo")
        inv_e_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_evo_prev")
        metric_choice_e = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=True, key="metric_evo")
        compare_e = st.checkbox("Comparar con año anterior (alineado por día)", value=False, key="cmp_evo")
        run_evo = st.button("Calcular evolución", type="primary", key="btn_evo")

    st.subheader("📉 Evolución de KPIs vs fecha de corte")
    help_block("Evolución por corte")
    if run_evo:
        cut_start_ts = pd.to_datetime(evo_cut_start)
        cut_end_ts = pd.to_datetime(evo_cut_end)
        if cut_start_ts > cut_end_ts:
            st.error("El inicio del rango de corte no puede ser posterior al fin.")
        else:
            rows_e = []
            for c in pd.date_range(cut_start_ts, cut_end_ts, freq='D'):
                _bp, tot_c = compute_kpis(
                    df_all=raw,
                    cutoff=c,
                    period_start=pd.to_datetime(evo_target_start),
                    period_end=pd.to_datetime(evo_target_end),
                    inventory_override=int(inv_e) if inv_e > 0 else None,
                    filter_props=props_e if props_e else None,
                )
                rows_e.append({"Corte": c.normalize(), **tot_c})
            df_evo = pd.DataFrame(rows_e)

            if df_evo.empty:
                st.info("No hay datos para el rango seleccionado.")
            else:
                key_col = METRIC_MAP[metric_choice_e]
                idx = pd.to_datetime(df_evo["Corte"])  # eje X con fechas reales
                plot_df = pd.DataFrame({"Actual": df_evo[key_col].values}, index=idx)

                if compare_e:
                    rows_prev = []
                    cut_start_prev = cut_start_ts - pd.DateOffset(years=1)
                    cut_end_prev = cut_end_ts - pd.DateOffset(years=1)
                    target_start_prev = pd.to_datetime(evo_target_start) - pd.DateOffset(years=1)
                    target_end_prev = pd.to_datetime(evo_target_end) - pd.DateOffset(years=1)
                    prev_dates = list(pd.date_range(cut_start_prev, cut_end_prev, freq='D'))
                    for c in prev_dates:
                        _bp2, tot_c2 = compute_kpis(
                            df_all=raw,
                            cutoff=c,
                            period_start=target_start_prev,
                            period_end=target_end_prev,
                            inventory_override=int(inv_e_prev) if inv_e_prev > 0 else None,
                            filter_props=props_e if props_e else None,
                        )
                        rows_prev.append(tot_c2[key_col])
                    prev_idx_aligned = pd.to_datetime(prev_dates) + pd.DateOffset(years=1)
                    s_prev = pd.Series(rows_prev, index=prev_idx_aligned)
                    plot_df["Año anterior"] = s_prev.reindex(idx).values

                st.line_chart(plot_df, height=300)
                st.dataframe(df_evo[["Corte", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]].rename(columns={
                    "noches_ocupadas": "Noches ocupadas",
                    "noches_disponibles": "Noches disponibles",
                    "ocupacion_pct": "Ocupación %",
                    "adr": "ADR (€)",
                    "revpar": "RevPAR (€)",
                    "ingresos": "Ingresos (€)"
                }), use_container_width=True)
                csve = df_evo.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Descargar evolución (CSV)", data=csve, file_name="evolucion_kpis.csv", mime="text/csv")
    else:
        st.caption("Configura los parámetros en la barra lateral, luego pulsa **Calcular evolución**.")

# =============================
# MODO 4: Pickup (entre dos cortes) — con Diario/Acumulado + Δ
# =============================
elif mode == "Pickup (entre dos cortes)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutA = st.date_input("Corte A", value=date(2024, 8, 14), key="pickup_cutA")
        cutB = st.date_input("Corte B", value=date(2024, 8, 21), key="pickup_cutB")
        c1, c2 = st.columns(2)
        p_start, p_end = period_inputs("Inicio del periodo", "Fin del periodo", date(2024, 9, 1), date(2024, 9, 30), "pickup")
        inv_pick = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="inv_pick")
        props_pick = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_pick")
        metric_pick = st.radio("Métrica gráfica", ["Noches", "Ingresos (€)", "Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=False)
        view_pick = st.radio("Vista", ["Diario", "Acumulado"], horizontal=True)
        topn = st.number_input("Top-N alojamientos (por pickup noches)", min_value=5, max_value=100, value=20, step=5)
        run_pick = st.button("Calcular pickup", type="primary")

    st.subheader("📈 Pickup entre cortes (B – A)")
    help_block("Pickup")
    if run_pick:
        if pd.to_datetime(cutA) > pd.to_datetime(cutB):
            st.error("Corte A no puede ser posterior a Corte B.")
        else:
            inv_override = int(inv_pick) if inv_pick > 0 else None
            # Totales A y B
            _bpA, totA = compute_kpis(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            _bpB, totB = compute_kpis(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            # Deltas totales
            deltas = {
                "noches": totB['noches_ocupadas'] - totA['noches_ocupadas'],
                "ingresos": totB['ingresos'] - totA['ingresos'],
                "occ_delta": totB['ocupacion_pct'] - totA['ocupacion_pct'],
                "adr_delta": totB['adr'] - totA['adr'],
                "revpar_delta": totB['revpar'] - totA['revpar'],
            }
            c1, c2, c3 = st.columns(3)
            c1.metric("Pickup Noches", f"{deltas['noches']:,}".replace(",", "."))
            c2.metric("Pickup Ingresos (€)", f"{deltas['ingresos']:.2f}")
            c3.metric("Δ Ocupación", f"{deltas['occ_delta']:.2f}%")
            c4, c5 = st.columns(2)
            c4.metric("Δ ADR", f"{deltas['adr_delta']:.2f}")
            c5.metric("Δ RevPAR", f"{deltas['revpar_delta']:.2f}")

            # Series diarias A y B
            serA = daily_series(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), props_pick if props_pick else None, inv_override)
            serB = daily_series(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), props_pick if props_pick else None, inv_override)
            # Elegir métrica
            key_map = {"Noches": "noches_ocupadas", "Ingresos (€)": "ingresos", "Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}
            k = key_map[metric_pick]
            df_plot = serA.merge(serB, on="Fecha", suffixes=(" A", " B"))
            df_plot["Δ (B–A)"] = df_plot[f"{k} B"] - df_plot[f"{k} A"]
            if view_pick == "Acumulado":
                for col in [f"{k} A", f"{k} B", "Δ (B–A)"]:
                    df_plot[col] = df_plot[col].cumsum()
            chart_df = pd.DataFrame({
                f"A (≤ {pd.to_datetime(cutA).date()})": df_plot[f"{k} A"].values,
                f"B (≤ {pd.to_datetime(cutB).date()})": df_plot[f"{k} B"].values,
                "Δ (B–A)": df_plot["Δ (B–A)"].values,
            }, index=pd.to_datetime(df_plot["Fecha"]))
            st.line_chart(chart_df, height=320)

            # Top-N alojamientos por pickup (noches e ingresos)
            bpA, _ = compute_kpis(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            bpB, _ = compute_kpis(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            merge = bpA.merge(bpB, on="Alojamiento", how="outer", suffixes=(" A", " B")).fillna(0)
            merge["Pickup noches"] = merge["Noches ocupadas B"] - merge["Noches ocupadas A"]
            merge["Pickup ingresos (€)"] = merge["Ingresos B"] - merge["Ingresos A"]
            top = merge.sort_values("Pickup noches", ascending=False).head(int(topn))
            st.subheader("🏆 Top alojamientos por pickup (noches)")
            st.dataframe(top[["Alojamiento", "Pickup noches", "Pickup ingresos (€)", "Noches ocupadas A", "Noches ocupadas B"]], use_container_width=True)

            csvp = df_plot.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar detalle pickup (CSV)", data=csvp, file_name="pickup_detalle.csv", mime="text/csv")
    else:
        st.caption("Configura parámetros y pulsa **Calcular pickup**.")

# =============================
# MODO 5: Pace (curva D-0…D-max)
# =============================
elif mode == "Pace (curva D)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        c1, c2 = st.columns(2)
        p_start, p_end = period_inputs("Inicio del periodo", "Fin del periodo", date(2024, 9, 1), date(2024, 9, 30), "pace")
        dmax = st.slider("D máximo (días antes)", min_value=30, max_value=365, value=120, step=10)
        props_p = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="pace_props")
        inv_p = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="pace_inv")
        metric_p = st.radio("Métrica", ["Ocupación %", "Noches", "Ingresos (€)", "ADR (€)", "RevPAR (€)"], horizontal=False)
        compare_yoy = st.checkbox("Comparar con año anterior", value=False)
        inv_p_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="pace_inv_prev")
        run_p = st.button("Calcular pace", type="primary")

    st.subheader("🏁 Pace: evolución hacia la estancia (D)")
    help_block("Pace")
    if run_p:
        base = pace_series(raw, pd.to_datetime(p_start), pd.to_datetime(p_end), int(dmax), props_p if props_p else None, int(inv_p) if inv_p > 0 else None)
        plot_df = base.copy()
        col = METRIC_MAP.get(metric_p, None)
        if metric_p == "Noches":
            y = "noches"
        elif metric_p == "Ingresos (€)":
            y = "ingresos"
        elif col is not None:
            y = col
        else:
            y = "noches"
        plot = pd.DataFrame({"Actual": plot_df[y].values}, index=plot_df["D"])  # eje X = D

        if compare_yoy:
            p_start_prev = pd.to_datetime(p_start) - pd.DateOffset(years=1)
            p_end_prev = pd.to_datetime(p_end) - pd.DateOffset(years=1)
            prev = pace_series(raw, p_start_prev, p_end_prev, int(dmax), props_p if props_p else None, int(inv_p_prev) if inv_p_prev > 0 else None)
            plot["Año anterior"] = prev[y].values
        st.line_chart(plot, height=320)
        st.dataframe(base, use_container_width=True)
        csvpace = base.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar pace (CSV)", data=csvpace, file_name="pace_curva.csv", mime="text/csv")
    else:
        st.caption("Configura parámetros y pulsa **Calcular pace**.")

# =============================
# MODO nuevo: Predicción (Pace)
# =============================
elif mode == "Predicción (Pace)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros de predicción")
        cut_f = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="f_cut")
        c1, c2 = st.columns(2)
        f_start, f_end = period_inputs("Inicio del periodo", "Fin del periodo", date(2024, 9, 1), date(2024, 9, 30), "forecast")
        ref_years = st.slider("Años de referencia (mismo mes)", min_value=1, max_value=3, value=2)
        dmax_f = st.slider("D máximo perfil", min_value=60, max_value=365, value=180, step=10)
        props_f = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="f_props")
        inv_f = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="f_inv")
        run_f = st.button("Calcular predicción", type="primary")

    st.subheader("🔮 Predicción mensual por Pace")
    help_block("Predicción")
    if run_f:
        res = pace_forecast_month(raw, pd.to_datetime(cut_f), pd.to_datetime(f_start), pd.to_datetime(f_end), int(ref_years), int(dmax_f), props_f if props_f else None, int(inv_f) if inv_f>0 else None)
        nights_otb = res['nights_otb']; nights_p25 = res['nights_p25']; nights_p50 = res['nights_p50']; nights_p75 = res['nights_p75']
        adr_final_p50 = res['adr_final_p50']; rev_final_p50 = res['revenue_final_p50']
        adr_tail_p25 = res['adr_tail_p25']; adr_tail_p50 = res['adr_tail_p50']; adr_tail_p75 = res['adr_tail_p75']
        pickup_needed = res['pickup_needed_p50']; pick_typ50 = res['pickup_typ_p50']; pick_typ75 = res['pickup_typ_p75']
        daily = res['daily'].copy()
        daily['OTB acumulado'] = daily['noches_ocupadas'].cumsum()

        # Tarjetas
        c1, c2, c3 = st.columns(3)
        c1.metric("OTB Noches", f"{nights_otb:,.0f}".replace(",","."))
        c2.metric("Forecast Noches (P50)", f"{nights_p50:,.0f}".replace(",","."))
        c3.metric("Forecast Ingresos (P50)", f"{rev_final_p50:,.2f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("ADR final (P50)", f"{adr_final_p50:,.2f}")
        low_band = min(nights_p25, nights_p75); high_band = max(nights_p25, nights_p75)
        c5.metric("Banda Noches [P25–P75]", f"[{low_band:,.0f} – {high_band:,.0f}]".replace(",","."))
        # Semáforo pickup
        if pickup_needed <= pick_typ50:
            status = "🟢"
            txt = "Pickup necesario dentro del típico (P50)"
        elif pickup_needed <= pick_typ75:
            status = "🟠"
            txt = "Pickup por encima del P50 pero ≤ P75 histórico"
        else:
            status = "🔴"
            txt = "Pickup por encima del P75 histórico"
        c6.metric("Pickup necesario", f"{pickup_needed:,.0f}".replace(",","."))
        st.caption(f"{status} {txt} · Típico P50≈ {pick_typ50:,.0f} · P75≈ {pick_typ75:,.0f}".replace(",","."))

        # Información ADR tail
        st.caption(f"ADR del remanente (histórico): P25≈ {adr_tail_p25:,.2f} · P50≈ {adr_tail_p50:,.2f} · P75≈ {adr_tail_p75:,.2f}")

        # Gráfico con banda y reglas horizontales usando Altair
        df_band = pd.DataFrame({
            'Fecha': daily['Fecha'],
            'low': low_band,
            'high': high_band,
        })
        base = alt.Chart(daily).encode(x=alt.X('Fecha:T', title='Fecha'))
        line = base.mark_line().encode(y=alt.Y('OTB acumulado:Q', title='Noches acumuladas'))
        band = alt.Chart(df_band).mark_area(opacity=0.15).encode(x='Fecha:T', y='low:Q', y2='high:Q')
        rule_p50 = alt.Chart(pd.DataFrame({'y':[nights_p50]})).mark_rule(strokeDash=[6,4]).encode(y='y:Q')
        rule_p25 = alt.Chart(pd.DataFrame({'y':[low_band]})).mark_rule(strokeDash=[2,4]).encode(y='y:Q')
        rule_p75 = alt.Chart(pd.DataFrame({'y':[high_band]})).mark_rule(strokeDash=[2,4]).encode(y='y:Q')
        chart = (band + line + rule_p25 + rule_p50 + rule_p75).properties(height=320)
        st.altair_chart(chart, use_container_width=True)

        csvf = daily.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar detalle diario (CSV)", data=csvf, file_name="forecast_pace_diario.csv", mime="text/csv")
    else:
        st.caption("Configura y pulsa **Calcular predicción**.")

# =============================
# MODO nuevo: Pipeline 90–180 días
# =============================
elif mode == "Pipeline 90–180 días":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_pl = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="pl_cut")
        pl_start = st.date_input("Inicio del horizonte", value=date(2024, 9, 1), key="pl_start")
        pl_end = st.date_input("Fin del horizonte", value=date(2024, 11, 30), key="pl_end")
        inv_pl = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="pl_inv")
        cmp_ly_pl = st.checkbox("Comparar con LY", value=False)
        inv_pl_ly = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="pl_inv_ly")
        run_pl = st.button("Calcular pipeline", type="primary")
    st.subheader("📆 Pipeline de OTB por día")
    if run_pl:
        inv_now = int(inv_pl) if inv_pl>0 else None
        ser = daily_series(raw, pd.to_datetime(cut_pl), pd.to_datetime(pl_start), pd.to_datetime(pl_end), None, inv_now)
        ser = ser.sort_values('Fecha')
        ser['Fecha'] = pd.to_datetime(ser['Fecha'])
        st.line_chart(ser.set_index('Fecha')[['noches_ocupadas','ingresos']].rename(columns={'noches_ocupadas':'Noches','ingresos':'Ingresos (€)'}), height=320)
        if cmp_ly_pl:
            ser_ly = daily_series(raw, pd.to_datetime(cut_pl) - pd.DateOffset(years=1), pd.to_datetime(pl_start) - pd.DateOffset(years=1), pd.to_datetime(pl_end) - pd.DateOffset(years=1), None, int(inv_pl_ly) if inv_pl_ly>0 else None)
            ser_ly['Fecha'] = pd.to_datetime(ser_ly['Fecha']) + pd.DateOffset(years=1)
            merge = ser.merge(ser_ly, on='Fecha', how='left', suffixes=('',' (prev)'))
            st.dataframe(merge, use_container_width=True)
        else:
            st.dataframe(ser, use_container_width=True)
        csvpl = ser.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar pipeline (CSV)", data=csvpl, file_name="pipeline_diario.csv", mime="text/csv")
    else:
        st.caption("Define horizonte y pulsa **Calcular pipeline**.")

# =============================
# MODO nuevo: Gap vs Target
# =============================
elif mode == "Gap vs Target":
    if raw is None:
        st.stop()
    tgts = st.session_state.get("targets_df")
    with st.sidebar:
        st.header("Parámetros")
        cut_gt = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="gt_cut")
        months_sel = st.multiselect("Meses (YYYY-MM)", options=sorted(pd.period_range(raw['Fecha entrada'].min().to_period('M'), raw['Fecha salida'].max().to_period('M')).astype(str).tolist()))
        inv_gt = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="gt_inv")
        run_gt = st.button("Calcular gaps", type="primary")
    st.subheader("🎯 Brecha a Objetivo (Targets)")
    if tgts is None:
        st.info("Carga un CSV de targets para usar esta vista.")
    elif run_gt and months_sel:
        rows = []
        for ym in months_sel:
            p = pd.Period(ym, freq='M')
            s, e = p.start_time, p.end_time
            _, real = compute_kpis(raw, pd.to_datetime(cut_gt), s, e, int(inv_gt) if inv_gt>0 else None, None)
            y, m = p.year, p.month
            trow = tgts[(tgts['year']==y) & (tgts['month']==m)]
            tgt_occ = float(trow['target_occ_pct'].iloc[0]) if not trow.empty and 'target_occ_pct' in tgts.columns else np.nan
            tgt_adr = float(trow['target_adr'].iloc[0]) if not trow.empty and 'target_adr' in tgts.columns else np.nan
            tgt_revpar = float(trow['target_revpar'].iloc[0]) if not trow.empty and 'target_revpar' in tgts.columns else np.nan
            gap_occ = tgt_occ - real['ocupacion_pct'] if not np.isnan(tgt_occ) else np.nan
            gap_adr = tgt_adr - real['adr'] if not np.isnan(tgt_adr) else np.nan
            gap_revpar = tgt_revpar - real['revpar'] if not np.isnan(tgt_revpar) else np.nan
            rows.append({"Mes": ym, "Occ Real %": real['ocupacion_pct'], "Occ Target %": tgt_occ, "Gap Occ p.p.": gap_occ, "ADR Real": real['adr'], "ADR Target": tgt_adr, "Gap ADR": gap_adr, "RevPAR Real": real['revpar'], "RevPAR Target": tgt_revpar, "Gap RevPAR": gap_revpar})
        df_gap = pd.DataFrame(rows).set_index('Mes')
        st.dataframe(df_gap, use_container_width=True)
        st.line_chart(df_gap[[c for c in df_gap.columns if 'Occ' in c]], height=280)

# =============================
# MODO extra: Pricing – Mapa eficiencia
# =============================
elif mode == "Pricing – Mapa eficiencia":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_px = st.date_input("Fecha de corte", value=date(2024,8,21), key="px_cut")
        p1, p2 = st.columns(2)
        with p1:
            px_start = st.date_input("Inicio periodo", value=date(2024,9,1), key="px_start")
        with p2:
            px_end = st.date_input("Fin periodo", value=date(2024,9,30), key="px_end")
        inv_px = st.number_input("Inventario (para Occ%)", min_value=0, value=0, step=1, key="px_inv")
        run_px = st.button("Ver mapa", type="primary")
    st.subheader("💸 Eficiencia diaria: ADR vs Ocupación%")
    if run_px:
        inv_now = get_inventory(raw, int(inv_px) if inv_px>0 else None)
        ser = daily_series(raw, pd.to_datetime(cut_px), pd.to_datetime(px_start), pd.to_datetime(px_end), None, inv_now)
        ser['Occ %'] = ser['noches_ocupadas'] / inv_now * 100.0
        ser['ADR día'] = np.where(ser['noches_ocupadas']>0, ser['ingresos']/ser['noches_ocupadas'], np.nan)
        st.scatter_chart(ser.set_index('Fecha')[['ADR día','Occ %']], height=320)
        st.dataframe(ser[['Fecha','noches_ocupadas','Occ %','ADR día','ingresos']], use_container_width=True)

# =============================
# MODO extra: Cohortes (Alta × Estancia)
# =============================
elif mode == "Cohortes (Alta × Estancia)":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        props_c = st.multiselect("Alojamientos (opcional)", options=sorted(raw['Alojamiento'].unique()), default=[], key="coh_props")
        run_c = st.button("Generar cohortes", type="primary")
    st.subheader("🧩 Cohortes: Mes de creación × Mes de llegada (reservas)")
    if run_c:
        dfc = raw.copy()
        if props_c:
            dfc = dfc[dfc['Alojamiento'].isin(props_c)]
        dfc = dfc.dropna(subset=['Fecha alta','Fecha entrada'])
        dfc['Mes alta'] = dfc['Fecha alta'].dt.to_period('M').astype(str)
        dfc['Mes llegada'] = dfc['Fecha entrada'].dt.to_period('M').astype(str)
        piv = pd.pivot_table(dfc, index='Mes alta', columns='Mes llegada', values='Alojamiento', aggfunc='count', fill_value=0)
        st.dataframe(piv, use_container_width=True)
        csvc = piv.reset_index().to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar cohortes (CSV)", data=csvc, file_name="cohortes_alta_estancia.csv", mime="text/csv")

# =============================
# MODO extra: Estacionalidad
# =============================
elif mode == "Estacionalidad":
    if raw is None:
        st.stop()

    # --- Controles
    with st.sidebar:
        st.header("Parámetros")
        dim = st.radio("Vista", ["Mes del año", "Día de la semana", "Día del mes"], horizontal=False)
        # años disponibles según noches de estancia (rango completo del dataset)
        y_min = min(pd.concat([raw["Fecha entrada"], raw["Fecha salida"]]).dt.year.dropna())
        y_max = max(pd.concat([raw["Fecha entrada"], raw["Fecha salida"]]).dt.year.dropna())
        years_opts = list(range(int(y_min), int(y_max) + 1)) if pd.notna(y_min) and pd.notna(y_max) else []
        years_sel = st.multiselect("Años a incluir", options=years_opts, default=years_opts, help="Filtra por año natural de la **fecha de estancia** (o de check-in si eliges 'Reservas').")
        base = st.radio("Base de conteo", ["Noches (estancia)", "Reservas (check-in)"])
        if base == "Noches (estancia)":
            met = st.radio("Métrica", ["Noches", "Ingresos (€)", "ADR"], horizontal=True)
        else:
            met = st.radio("Métrica", ["Reservas"], horizontal=True)
        show_idx = st.checkbox("Mostrar índice (normalizado a media=1)", value=True)
        run_s = st.button("Calcular", type="primary")

    st.subheader("🍂 Estacionalidad – distribución por periodo")
    help_block("Estacionalidad")

    def _nightly_rows(df_all: pd.DataFrame, years: list[int]) -> pd.DataFrame:
        df = df_all.dropna(subset=["Fecha entrada", "Fecha salida", "Precio"]).copy()
        rows = []
        for _, r in df.iterrows():
            e, s, price = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
            n = (s - e).days
            if n <= 0:
                continue
            adr_n = price / n if n > 0 else 0.0
            for d in pd.date_range(e, s - pd.Timedelta(days=1), freq='D'):
                if years and d.year not in years:
                    continue
                rows.append({
                    "Fecha": d.normalize(),
                    "Año": d.year,
                    "MesN": d.month,
                    "Mes": {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}[d.month],
                    "DOW": ("Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo")[d.weekday()],
                    "DOM": d.day,
                    "Noches": 1,
                    "Ingresos": adr_n,
                })
        return pd.DataFrame(rows)

    if run_s:
        if base == "Noches (estancia)":
            nights_df = _nightly_rows(raw, years_sel)
            if nights_df.empty:
                st.info("No hay noches en el filtro seleccionado.")
            else:
                if dim == "Mes del año":
                    g = nights_df.groupby(["Mes", "MesN"]).agg(Noches=("Noches","sum"), Ingresos=("Ingresos","sum")).reset_index().sort_values("MesN")
                    g["ADR"] = np.where(g["Noches"]>0, g["Ingresos"]/g["Noches"], np.nan)
                    order = g["Mes"].values
                    if met == "Noches":
                        vals = g[["Mes","Noches"]].set_index("Mes")
                    elif met == "Ingresos (€)":
                        vals = g[["Mes","Ingresos"]].set_index("Mes")
                    else:
                        vals = g[["Mes","ADR"]].set_index("Mes")
                elif dim == "Día de la semana":
                    g = nights_df.groupby("DOW").agg(Noches=("Noches","sum"), Ingresos=("Ingresos","sum")).reset_index()
                    g["ADR"] = np.where(g["Noches"]>0, g["Ingresos"]/g["Noches"], np.nan)
                    dow_order = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
                    g = g.set_index("DOW").reindex(dow_order).reset_index()
                    order = g["DOW"].values
                    if met == "Noches":
                        vals = g[["DOW","Noches"]].set_index("DOW")
                    elif met == "Ingresos (€)":
                        vals = g[["DOW","Ingresos"]].set_index("DOW")
                    else:
                        vals = g[["DOW","ADR"]].set_index("DOW")
                else:  # Día del mes
                    g = nights_df.groupby("DOM").agg(Noches=("Noches","sum"), Ingresos=("Ingresos","sum")).reset_index()
                    g["ADR"] = np.where(g["Noches"]>0, g["Ingresos"]/g["Noches"], np.nan)
                    order = g["DOM"].values
                    if met == "Noches":
                        vals = g.set_index("DOM")["Noches"].to_frame()
                    elif met == "Ingresos (€)":
                        vals = g.set_index("DOM")["Ingresos"].to_frame()
                    else:
                        vals = g.set_index("DOM")["ADR"].to_frame()

                if show_idx and met != "ADR":
                    # ADR no se normaliza (ya es ratio). Para Noches/Ingresos hacemos índice relativo a la media
                    serie = vals.iloc[:,0]
                    idx = serie / (serie.mean() if serie.mean()!=0 else 1)
                    out = pd.DataFrame({vals.columns[0]: serie, "Índice": idx})
                    st.line_chart(idx.rename("Índice"))
                    st.dataframe(out.reset_index().rename(columns={"index": dim}), use_container_width=True)
                else:
                    st.line_chart(vals)
                    st.dataframe(vals.reset_index().rename(columns={"index": dim, vals.columns[0]: met}), use_container_width=True)

        else:  # Reservas por check-in
            dfr = raw.dropna(subset=["Fecha entrada"]).copy()
            dfr["Año"] = dfr["Fecha entrada"].dt.year
            if years_sel:
                dfr = dfr[dfr["Año"].isin(years_sel)]
            if dfr.empty:
                st.info("No hay reservas en el filtro seleccionado.")
            else:
                dfr["Mes"] = dfr["Fecha entrada"].dt.month.map({1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'})
                dfr["MesN"] = dfr["Fecha entrada"].dt.month
                dfr["DOW"] = dfr["Fecha entrada"].dt.weekday.map({0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"})
                dfr["DOM"] = dfr["Fecha entrada"].dt.day
                if dim == "Mes del año":
                    g = dfr.groupby(["Mes","MesN"]).size().reset_index(name="Reservas").sort_values("MesN")
                    vals = g.set_index("Mes")["Reservas"].to_frame()
                elif dim == "Día de la semana":
                    g = dfr.groupby("DOW").size().reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]).fillna(0).astype(int)
                    vals = g.to_frame(name="Reservas")
                else:
                    g = dfr.groupby("DOM").size()
                    vals = g.to_frame(name="Reservas").sort_index()
                if show_idx:
                    serie = vals.iloc[:,0]
                    idx = serie / (serie.mean() if serie.mean()!=0 else 1)
                    st.line_chart(idx.rename("Índice"))
                    out = pd.DataFrame({"Valor": serie, "Índice": idx})
                    st.dataframe(out.reset_index().rename(columns={"index": dim, "Valor": "Reservas"}), use_container_width=True)
                else:
                    st.line_chart(vals)
                    st.dataframe(vals.reset_index().rename(columns={"index": dim}), use_container_width=True)

# =============================
# MODO extra: Ranking alojamientos
# =============================
elif mode == "Ranking alojamientos":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_rk = st.date_input("Fecha de corte", value=date(2024,8,21), key="rk_cut")
        r1, r2 = st.columns(2)
        with r1:
            rk_start = st.date_input("Inicio periodo", value=date(2024,9,1), key="rk_start")
        with r2:
            rk_end = st.date_input("Fin periodo", value=date(2024,9,30), key="rk_end")
        run_rk = st.button("Calcular ranking", type="primary")
    st.subheader("🏅 Ranking de alojamientos")
    if run_rk:
        bp, tot = compute_kpis(raw, pd.to_datetime(cut_rk), pd.to_datetime(rk_start), pd.to_datetime(rk_end), None, None)
        if bp.empty:
            st.info("Sin datos en el rango.")
        else:
            bp['RevPAR estim.'] = bp['Ingresos'] / ((pd.to_datetime(rk_end)-pd.to_datetime(rk_start)).days + 1)
            st.dataframe(bp.sort_values('Ingresos', ascending=False), use_container_width=True)
            csvrk = bp.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 Descargar ranking (CSV)", data=csvrk, file_name="ranking_alojamientos.csv", mime="text/csv")

# =============================
# MODO extra: Operativa
# =============================
elif mode == "Operativa":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_op = st.date_input("Fecha de corte", value=date(2024,8,21), key="op_cut")
        o1, o2 = st.columns(2)
        with o1:
            op_start = st.date_input("Inicio periodo", value=date(2024,9,1), key="op_start")
        with o2:
            op_end = st.date_input("Fin periodo", value=date(2024,9,30), key="op_end")
        inv_op = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="op_inv")
        run_op = st.button("Calcular operativa", type="primary")
    st.subheader("🛎️ Operativa diaria")
    if run_op:
        inv_now = get_inventory(raw, int(inv_op) if inv_op>0 else None)
        dfc = raw[raw['Fecha alta'] <= pd.to_datetime(cut_op)].copy()
        days = pd.date_range(pd.to_datetime(op_start), pd.to_datetime(op_end), freq='D')
        chk_in = dfc['Fecha entrada'].dt.normalize().value_counts()
        chk_out = dfc['Fecha salida'].dt.normalize().value_counts()
        active = daily_series(raw, pd.to_datetime(cut_op), pd.to_datetime(op_start), pd.to_datetime(op_end), None, inv_now)
        out = pd.DataFrame({'Fecha': days})
        out['Check-ins'] = out['Fecha'].map(chk_in).fillna(0).astype(int)
        out['Check-outs'] = out['Fecha'].map(chk_out).fillna(0).astype(int)
        out = out.merge(active[['Fecha','noches_ocupadas']], on='Fecha', how='left').rename(columns={'noches_ocupadas':'Estancias activas'})
        out['Capacidad restante'] = inv_now - out['Estancias activas']
        out = out.fillna(0)
        st.dataframe(out, use_container_width=True)
        st.line_chart(out.set_index('Fecha')[['Estancias activas','Capacidad restante']], height=300)

# =============================
# MODO extra: Calidad de datos
# =============================
elif mode == "Calidad de datos":
    if raw is None:
        st.stop()
    st.subheader("🔧 Chequeo de datos")
    dfq = raw.copy()
    issues = []
    # Fechas incoherentes
    bad_dates = dfq[(dfq['Fecha salida'] <= dfq['Fecha entrada']) | (dfq['Fecha entrada'].isna()) | (dfq['Fecha salida'].isna())]
    if not bad_dates.empty:
        st.warning(f"Fechas incoherentes: {len(bad_dates)} filas")
        st.dataframe(bad_dates, use_container_width=True)
    # Precio <= 0
    bad_price = dfq[(pd.to_numeric(dfq['Precio'], errors='coerce').fillna(0) <= 0)]
    if not bad_price.empty:
        st.warning(f"Precios nulos/negativos: {len(bad_price)} filas")
        st.dataframe(bad_price, use_container_width=True)
    # LOS 0
    dfq['los'] = (dfq['Fecha salida'].dt.normalize() - dfq['Fecha entrada'].dt.normalize()).dt.days
    los0 = dfq[dfq['los'] <= 0]
    if not los0.empty:
        st.warning(f"LOS ≤ 0: {len(los0)} filas")
        st.dataframe(los0, use_container_width=True)

# =============================
# MODO 6: Lead time & LOS
# =============================
elif mode == "Lead time & LOS":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        c1, c2 = st.columns(2)
        lt_start, lt_end = period_inputs("Inicio periodo (por llegada)", "Fin periodo (por llegada)", date(2024, 9, 1), date(2024, 9, 30), "lt")
        props_lt = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="lt_props")
        run_lt = st.button("Calcular", type="primary")

    st.subheader("⏱️ Lead time (por reserva) y LOS")
    help_block("Lead")
    if run_lt:
        df = raw.copy()
        if props_lt:
            df = df[df["Alojamiento"].isin(props_lt)]
        df = df.dropna(subset=["Fecha alta", "Fecha entrada", "Fecha salida"]) 
        # Filtro por llegada en periodo
        mask = (df["Fecha entrada"] >= pd.to_datetime(lt_start)) & (df["Fecha entrada"] <= pd.to_datetime(lt_end))
        df = df[mask]
        if df.empty:
            st.info("Sin reservas en el rango seleccionado.")
        else:
            df["lead_days"] = (df["Fecha entrada"].dt.normalize() - df["Fecha alta"].dt.normalize()).dt.days.clip(lower=0)
            df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
            # Percentiles Lead
            pcts = [50, 75, 90]
            lt_p = {f"P{p}": np.percentile(df["lead_days"], p) for p in pcts}
            los_p = {f"P{p}": np.percentile(df["los"], p) for p in pcts}
            c1, c2, c3 = st.columns(3)
            c1.metric("Lead medio (d)", f"{df['lead_days'].mean():.1f}")
            c2.metric("LOS medio (noches)", f"{df['los'].mean():.1f}")
            c3.metric("Lead mediana (d)", f"{np.percentile(df['lead_days'],50):.0f}")

            # Histogramas como tablas (conteos por bins estándar)
            lt_bins = [0,3,7,14,30,60,120,1e9]
            los_bins = [1,2,3,4,5,7,10,14,21,30, np.inf]
            lt_labels = ["0-3","4-7","8-14","15-30","31-60","61-120","120+"]
            los_labels = ["1","2","3","4","5-7","8-10","11-14","15-21","22-30","30+"]
            lt_cat = pd.cut(df["lead_days"], bins=lt_bins, labels=lt_labels, right=True)
            los_cat = pd.cut(df["los"], bins=los_bins, labels=los_labels, right=True, include_lowest=True)
            lt_tab = lt_cat.value_counts().reindex(lt_labels).fillna(0).astype(int).rename_axis("Lead bin").reset_index(name="Reservas")
            los_tab = los_cat.value_counts().reindex(los_labels).fillna(0).astype(int).rename_axis("LOS bin").reset_index(name="Reservas")
            st.markdown("**Lead time (reservas)**")
            st.dataframe(lt_tab, use_container_width=True)
            st.markdown("**LOS (reservas)**")
            st.dataframe(los_tab, use_container_width=True)
            csv_lt = lt_tab.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar Lead bins (CSV)", data=csv_lt, file_name="lead_bins.csv", mime="text/csv")
            csv_los = los_tab.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar LOS bins (CSV)", data=csv_los, file_name="los_bins.csv", mime="text/csv")
    else:
        st.caption("Elige el rango de llegada y pulsa **Calcular**.")

# =============================
# MODO 7: DOW heatmap (periodo)
# =============================
elif mode == "DOW heatmap":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        c1, c2 = st.columns(2)
        h_start, h_end = period_inputs("Inicio periodo", "Fin periodo", date(2024, 9, 1), date(2024, 9, 30), "dow")
        props_h = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="dow_props")
        mode_h = st.radio("Métrica", ["Ocupación (noches)", "Ocupación (%)", "ADR (€)"], horizontal=True)
        inv_h = st.number_input("Inventario (para %)", min_value=0, value=0, step=1, key="dow_inv")
        cutoff_h = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="dow_cutoff")
        run_h = st.button("Generar heatmap", type="primary")

    st.subheader("🗓️ Heatmap Día de la Semana × Mes")
    help_block("DOW")
    if run_h:
        df_cut = raw[raw["Fecha alta"] <= pd.to_datetime(cutoff_h)].copy()
        if props_h:
            df_cut = df_cut[df_cut["Alojamiento"].isin(props_h)]
        df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]) 
        rows = []
        for _, r in df_cut.iterrows():
            e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
            ov_start = max(e, pd.to_datetime(h_start))
            ov_end = min(s, pd.to_datetime(h_end) + pd.Timedelta(days=1))
            n_nights = (s - e).days
            if ov_start >= ov_end or n_nights <= 0:
                continue
            adr_night = p / n_nights if n_nights > 0 else 0.0
            for d in pd.date_range(ov_start, ov_end - pd.Timedelta(days=1), freq='D'):
                rows.append({"Mes": d.strftime('%Y-%m'), "DOW": ("Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo")[d.weekday()], "Noches": 1, "ADR": adr_night, "Fecha": d.normalize()})
        if not rows:
            st.info("Sin datos en el rango.")
        else:
            df_n = pd.DataFrame(rows)
            if mode_h == "Ocupación (noches)":
                piv = df_n.pivot_table(index="DOW", columns="Mes", values="Noches", aggfunc='sum', fill_value=0)
                st.dataframe(piv.reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]), use_container_width=True)
            elif mode_h == "Ocupación (%)":
                inv_now = get_inventory(raw, int(inv_h) if inv_h>0 else None)
                occ = occurrences_of_dow_by_month(pd.to_datetime(h_start), pd.to_datetime(h_end))
                nights_piv = df_n.pivot_table(index="DOW", columns="Mes", values="Noches", aggfunc='sum', fill_value=0)
                # Calcular % = noches / (inv * #ocurrencias de ese DOW en ese mes)
                out_cols = {}
                for mes in nights_piv.columns:
                    for dow in nights_piv.index:
                        n_occ = occ[(occ['Mes']==mes) & (occ['DOW']==dow)]['occ']
                        denom = (inv_now * (int(n_occ.iloc[0]) if not n_occ.empty else 0))
                        val = nights_piv.loc[dow, mes] / denom * 100.0 if denom>0 else 0.0
                        out_cols.setdefault(mes, {})[dow] = val
                pivp = pd.DataFrame(out_cols)
                pivp = pivp.reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"])
                st.dataframe(pivp, use_container_width=True)
            else:
                piv = df_n.pivot_table(index="DOW", columns="Mes", values="ADR", aggfunc='mean', fill_value=0.0)
                st.dataframe(piv.reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]), use_container_width=True)
            csvh = (piv if mode_h!="Ocupación (%)" else pivp).reset_index().to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar heatmap (CSV)", data=csvh, file_name="dow_heatmap.csv", mime="text/csv")
    else:
        st.caption("Elige periodo y pulsa **Generar heatmap**.")

# =============================
# MODO 8: ADR bands & Targets
# =============================
elif mode == "ADR bands & Targets":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros ADR bands")
        ab_cutoff = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="ab_cutoff")
        c1, c2 = st.columns(2)
        ab_start, ab_end = period_inputs("Inicio periodo", "Fin periodo", date(2024, 9, 1), date(2024, 9, 30), "ab")
        props_ab = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="ab_props")
        run_ab = st.button("Calcular ADR bands", type="primary")

    st.subheader("📦 Bandas de ADR (percentiles por mes)")
    help_block("ADR bands")
    if run_ab:
        df = raw[raw["Fecha alta"] <= pd.to_datetime(ab_cutoff)].copy()
        if props_ab:
            df = df[df["Alojamiento"].isin(props_ab)]
        df = df.dropna(subset=["Fecha entrada", "Fecha salida"]) 
        df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
        df["adr_reserva"] = df["Precio"] / df["los"]
        # Filtrar por periodo (por estancia que intersecta)
        ov_start = pd.to_datetime(ab_start)
        ov_end = pd.to_datetime(ab_end) + pd.Timedelta(days=1)
        mask = ~((df["Fecha salida"] <= ov_start) | (df["Fecha entrada"] >= ov_end))
        df = df[mask]
        if df.empty:
            st.info("Sin reservas en el rango.")
        else:
            df["Mes"] = df["Fecha entrada"].dt.to_period('M').astype(str)
            def pct_cols(x):
                arr = x.dropna().values
                if arr.size == 0:
                    return pd.Series({"P10": 0.0, "P25": 0.0, "Mediana": 0.0, "P75": 0.0, "P90": 0.0})
                return pd.Series({
                    "P10": np.percentile(arr, 10),
                    "P25": np.percentile(arr, 25),
                    "Mediana": np.percentile(arr, 50),
                    "P75": np.percentile(arr, 75),
                    "P90": np.percentile(arr, 90),
                })
            bands = df.groupby("Mes")["adr_reserva"].apply(pct_cols).reset_index()
            bands_wide = bands.pivot(index="Mes", columns="level_1", values="adr_reserva").sort_index()
            st.dataframe(bands_wide, use_container_width=True)
            # Gráfica de P10/Mediana/P90 + ADR OTB
            # Calculamos ADR OTB por mes para añadirlo como serie
            adr_otb_map = {}
            months_in_view = bands_wide.index.tolist()
            for ym in months_in_view:
                p = pd.Period(ym, freq='M')
                m_start, m_end = p.start_time, p.end_time
                _bp_m, tot_m = compute_kpis(raw, pd.to_datetime(ab_cutoff), m_start, m_end, None, props_ab if props_ab else None)
                adr_otb_map[ym] = float(tot_m['adr'])

            plot = bands_wide[["P10","Mediana","P90"]].copy()
            plot["ADR OTB"] = [adr_otb_map.get(ym, np.nan) for ym in plot.index]
            st.line_chart(plot, height=300)

            # ▶️ ADR OTB actual y posición en la banda (aprox)
            months_in_view = bands_wide.index.tolist()
            rows_badge = []
            for ym in months_in_view:
                p = pd.Period(ym, freq='M')
                m_start, m_end = p.start_time, p.end_time
                _bp_m, tot_m = compute_kpis(raw, pd.to_datetime(ab_cutoff), m_start, m_end, None, props_ab if props_ab else None)
                adr_otb_m = float(tot_m['adr'])
                q10 = float(bands_wide.loc[ym, 'P10']); q25 = float(bands_wide.loc[ym, 'P25']); q50 = float(bands_wide.loc[ym, 'Mediana']); q75 = float(bands_wide.loc[ym, 'P75']); q90 = float(bands_wide.loc[ym, 'P90'])
                def interp_pct(v, q10,q25,q50,q75,q90):
                    try:
                        if v <= q10: return max(0.0, 10.0 * (v / q10)) if q10>0 else 0.0
                        if v <= q25: return 10.0 + (v - q10) / max(q25-q10,1e-9) * 15.0
                        if v <= q50: return 25.0 + (v - q25) / max(q50-q25,1e-9) * 25.0
                        if v <= q75: return 50.0 + (v - q50) / max(q75-q50,1e-9) * 25.0
                        if v <= q90: return 75.0 + (v - q75) / max(q90-q75,1e-9) * 15.0
                        return 95.0  # ≥ P90
                    except Exception:
                        return np.nan
                p_est = interp_pct(adr_otb_m, q10,q25,q50,q75,q90)
                rows_badge.append({"Mes": ym, "ADR OTB (€)": round(adr_otb_m,2), "Posición banda (≈Pxx)": (f"P{int(round(p_est))}" if np.isfinite(p_est) else "–")})
            if rows_badge:
                st.markdown("**ADR actual vs banda (aprox.)**")
                st.dataframe(pd.DataFrame(rows_badge), use_container_width=True)

            csvb = bands_wide.reset_index().to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar ADR bands (CSV)", data=csvb, file_name="adr_bands.csv", mime="text/csv")

    st.divider()
    st.subheader("🎯 Targets vs Real vs LY (opcional)")
    tgts = st.session_state.get("targets_df")
    if tgts is None:
        st.info("Carga un CSV de targets en la barra lateral (dentro del acordeón 🎯).")
    else:
        with st.sidebar:
            t_cutoff = st.date_input("Fecha de corte para 'Real'", value=date(2024, 8, 21), key="tgt_cutoff")
            months_sel = st.multiselect("Meses (YYYY-MM)", options=sorted(tgts.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}", axis=1).unique().tolist()))
            inv_now = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="tgt_inv")
            inv_ly = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="tgt_inv_ly")
        if months_sel:
            rows = []
            for ym in months_sel:
                y, m = map(int, ym.split('-'))
                p = pd.Period(ym, freq='M')
                p_start = p.to_timestamp(how='start')
                p_end = p.to_timestamp(how='end')
                # Real
                _bp, real = compute_kpis(raw, pd.to_datetime(t_cutoff), p_start, p_end, int(inv_now) if inv_now>0 else None, None)
                # LY
                p_prev = p - 12
                _bp2, ly = compute_kpis(raw, pd.to_datetime(t_cutoff) - pd.DateOffset(years=1), p_prev.to_timestamp('M', 'start'), p_prev.to_timestamp('M', 'end'), int(inv_ly) if inv_ly>0 else None, None)
                # Target
                trow = tgts[(tgts['year']==y) & (tgts['month']==m)]
                tgt_occ = float(trow['target_occ_pct'].iloc[0]) if 'target_occ_pct' in tgts.columns and not trow.empty else np.nan
                tgt_adr = float(trow['target_adr'].iloc[0]) if 'target_adr' in tgts.columns and not trow.empty else np.nan
                tgt_revpar = float(trow['target_revpar'].iloc[0]) if 'target_revpar' in tgts.columns and not trow.empty else np.nan
                rows.append({
                    "Mes": ym,
                    "Occ Real %": real['ocupacion_pct'],
                    "Occ LY %": ly['ocupacion_pct'],
                    "Occ Target %": tgt_occ,
                    "ADR Real": real['adr'],
                    "ADR LY": ly['adr'],
                    "ADR Target": tgt_adr,
                    "RevPAR Real": real['revpar'],
                    "RevPAR LY": ly['revpar'],
                    "RevPAR Target": tgt_revpar,
                })
            df_t = pd.DataFrame(rows).set_index("Mes")
            st.dataframe(df_t, use_container_width=True)
            st.line_chart(df_t[["Occ Real %","Occ LY %","Occ Target %"]].dropna(), height=280)

# =============================
# MODO 9: Calendario por alojamiento (heatmap simple)
# =============================
elif mode == "Calendario por alojamiento":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_cal = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cal_cutoff")
        c1, c2 = st.columns(2)
        cal_start, cal_end = period_inputs("Inicio periodo", "Fin periodo", date(2024, 9, 1), date(2024, 9, 30), "cal")
        props_cal = st.multiselect("Alojamientos", options=sorted(raw["Alojamiento"].unique()), default=[], key="cal_props")
        mode_cal = st.radio("Modo", ["Ocupado/Libre", "ADR"], horizontal=True, key="cal_mode")
        run_cal = st.button("Generar calendario", type="primary", key="btn_cal")

    st.subheader("🗓️ Calendario por alojamiento")
    help_block("Calendario")
    if run_cal:
        if pd.to_datetime(cal_start) > pd.to_datetime(cal_end):
            st.error("El inicio del periodo no puede ser posterior al fin.")
        else:
            piv = build_calendar_matrix(
                df_all=raw,
                cutoff=pd.to_datetime(cutoff_cal),
                start=pd.to_datetime(cal_start),
                end=pd.to_datetime(cal_end),
                props=props_cal if props_cal else None,
                mode=mode_cal,
            )
            if piv.empty:
                st.info("Sin datos para los filtros seleccionados.")
            else:
                piv.columns = [c.strftime('%Y-%m-%d') if isinstance(c, (pd.Timestamp, datetime, date)) else str(c) for c in piv.columns]
                st.dataframe(piv, use_container_width=True)
                csvc = piv.reset_index().to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Descargar calendario (CSV)", data=csvc, file_name="calendario_alojamientos.csv", mime="text/csv")
    else:
        st.caption("Elige parámetros y pulsa **Generar calendario**.")

elif mode == "Resumen & Simulador":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_r = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cut_resumen")
        # Usa el periodo global si el interruptor está activo
        start_r, end_r = period_inputs("Inicio del periodo", "Fin del periodo", date(2024, 9, 1), date(2024, 9, 30), "resumen")
        props_r = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_resumen")
        inv_r = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="inv_resumen")
        inv_r_prev = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="inv_resumen_prev")
        ref_years_r = st.slider("Años de referencia (pace)", 1, 3, 2)
        dmax_r = st.slider("D máximo pace", 60, 365, 180, 10)
        st.markdown("—")
        st.subheader("Simulador")
        delta_price = st.slider("Ajuste ADR del remanente (%)", -30, 30, 0, 1, help="Aplica sólo al tramo aún por vender")
        elasticity = st.slider("Elasticidad de demanda (negativa)", -1.5, -0.2, -0.8, 0.1, help="Ej. -0.8: +10% precio ⇒ -8% noches del remanente")
        run_r = st.button("Calcular resumen", type="primary")

    st.subheader("📊 Resumen & Simulador")
    help_block("Resumen")  # (si no te aparece, añade el texto en el bloque de ayudas de abajo)

    if run_r:
        # --- Filtros base
        props_sel = props_r if props_r else None
        inv_now = int(inv_r) if inv_r > 0 else None

        # --- KPIs OTB actuales
        byp, tot = compute_kpis(raw, pd.to_datetime(cutoff_r), pd.to_datetime(start_r), pd.to_datetime(end_r), inv_now, props_sel)
        noches_otb = tot["noches_ocupadas"]
        ingresos_otb = tot["ingresos"]
        adr_otb = tot["adr"]

        # --- KPIs LY (corte y periodo -1 año)
        cutoff_ly = pd.to_datetime(cutoff_r) - pd.DateOffset(years=1)
        start_ly = pd.to_datetime(start_r) - pd.DateOffset(years=1)
        end_ly = pd.to_datetime(end_r) - pd.DateOffset(years=1)
        inv_ly = int(inv_r_prev) if inv_r_prev > 0 else None
        _, tot_ly = compute_kpis(raw, cutoff_ly, start_ly, end_ly, inv_ly, props_sel)

        # --- Pace (curva D) -> hitos D60/D30/D14 vs LY
        base_pace = pace_series(raw, pd.to_datetime(start_r), pd.to_datetime(end_r), int(dmax_r), props_sel, inv_now)
        ly_pace = pace_series(raw, start_ly, end_ly, int(dmax_r), props_sel, inv_ly)
        def val_at(df, D, col):
            if df.empty: return np.nan
            row = df.loc[df["D"]==int(D)]
            return float(row[col].values[0]) if len(row) else np.nan
        anchors = [60, 30, 14]
        pace_rows = []
        for D in anchors:
            occ_now = val_at(base_pace, D, "ocupacion_pct")
            occ_ly  = val_at(ly_pace,   D, "ocupacion_pct")
            delta = (occ_now - occ_ly) if (np.isfinite(occ_now) and np.isfinite(occ_ly)) else np.nan
            pace_rows.append({"Hito": f"D-{D}", "Occ% actual": occ_now, "Occ% LY": occ_ly, "Δ pp": delta})

        # --- ADR bands del periodo (por reservas que intersectan)
        dfb = raw[raw["Fecha alta"] <= pd.to_datetime(cutoff_r)].copy()
        if props_sel:
            dfb = dfb[dfb["Alojamiento"].isin(props_sel)]
        dfb = dfb.dropna(subset=["Fecha entrada","Fecha salida"])
        los = (dfb["Fecha salida"].dt.normalize() - dfb["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
        dfb["adr_reserva"] = (dfb["Precio"] / los)
        ov_start = pd.to_datetime(start_r); ov_end = pd.to_datetime(end_r) + pd.Timedelta(days=1)
        mask = ~((dfb["Fecha salida"] <= ov_start) | (dfb["Fecha entrada"] >= ov_end))
        dfb = dfb[mask]
        if dfb.empty:
            q10=q25=q50=q75=q90=np.nan
        else:
            arr = dfb["adr_reserva"].dropna().values
            q10,q25,q50,q75,q90 = [np.percentile(arr, p) for p in (10,25,50,75,90)]

        def pos_percentil(v, q10,q25,q50,q75,q90):
            if not np.isfinite(v): return np.nan
            try:
                if v <= q10: return max(0.0, 10.0*(v/max(q10,1e-9)))
                if v <= q25: return 10.0 + (v-q10)/max(q25-q10,1e-9)*15.0
                if v <= q50: return 25.0 + (v-q25)/max(q50-q25,1e-9)*25.0
                if v <= q75: return 50.0 + (v-q50)/max(q75-q50,1e-9)*25.0
                if v <= q90: return 75.0 + (v-q75)/max(q90-q75,1e-9)*15.0
                return 95.0
            except Exception:
                return np.nan
        pos_band = pos_percentil(adr_otb, q10,q25,q50,q75,q90)

        # --- DOW del periodo
        df_cut = raw[raw["Fecha alta"] <= pd.to_datetime(cutoff_r)].copy()
        if props_sel:
            df_cut = df_cut[df_cut["Alojamiento"].isin(props_sel)]
        df_cut = df_cut.dropna(subset=["Fecha entrada","Fecha salida"])
        rows_dow = []
        for _, r in df_cut.iterrows():
            e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
            ov_s = max(e, pd.to_datetime(start_r)); ov_e = min(s, pd.to_datetime(end_r) + pd.Timedelta(days=1))
            n = (s - e).days
            if ov_s >= ov_e or n<=0: continue
            adr_n = p / n
            for d in pd.date_range(ov_s, ov_e - pd.Timedelta(days=1), freq="D"):
                rows_dow.append({"dow": d.weekday(), "ADR": adr_n})
        dow_tab = pd.DataFrame(rows_dow)
        top_dow_txt = "—"
        if not dow_tab.empty:
            agg = dow_tab.groupby("dow").agg(Noches=("ADR","count"), ADR=("ADR","mean")).reset_index()
            agg["DOW"] = agg["dow"].map({0:"Lun",1:"Mar",2:"Mié",3:"Jue",4:"Vie",5:"Sáb",6:"Dom"})
            agg = agg.sort_values("Noches", ascending=False)
            top_dow_txt = f"Top noches: {', '.join(agg.head(2)['DOW'].tolist())} · Peor: {', '.join(agg.tail(2)['DOW'].tolist())}"

        # --- Inventario y noches disponibles del periodo
        inv_used = inv_now if inv_now else raw["Alojamiento"].nunique()
        dias = (pd.to_datetime(end_r) - pd.to_datetime(start_r)).days + 1
        noches_disp = inv_used * max(dias, 0)

        # --- Semáforos / Recomendación
        # Pace: cuantos hitos en positivo vs LY
        pos_hitos = sum(1 for r in pace_rows if np.isfinite(r["Δ pp"]) and r["Δ pp"] >= 0)
        if pos_hitos == 3: pace_flag = "🟢"