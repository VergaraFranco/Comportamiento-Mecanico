"""
Simulador de Comportamiento Mecánico de Materiales — v2
=========================================================
Curvas armónicas (sin tramos rectos artificiales):
  - Zona plástica uniforme: ecuación de Ludwik/Hollomon (σ = σy + K·εp^n)
  - Post-estricción: decaimiento suave (smoothstep) en tensión convencional
    y corrección por triaxialidad (tipo Bridgman) en tensión real.
  - Charpy: tanh de alta resolución (BCC), curva de saturación (FCC),
    y meseta baja (martensíticos).

"MIRAR + CONOCIMIENTO TÉCNICO = VER"
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(page_title="Simulador Mecánico de Materiales", page_icon="🔩", layout="wide")

E_MOD = 210_000.0     # Módulo de Young, MPa (aceros)
C_REF = 0.20          # %C de referencia de los grados ASME base
N_POINTS = 500        # resolución fina para curvas armónicas

# Ejes fijos para comparación visual entre materiales/condiciones
SIGMA_EPS_XRANGE = [0.0, 0.6]      # deformación
SIGMA_EPS_YRANGE = [0.0, 1000.0]   # MPa
CHARPY_XRANGE = [-200.0, 300.0]    # °C
CHARPY_YRANGE = [0.0, 350.0]       # J
TTRANS_REF_J = 20.0                 # energía que define la Temperatura de Transición
T_CREEP = 370.0                      # °C, umbral de daño por fluencia lenta

# ----------------------------------------------------------------------
# BASES DE DATOS
# ----------------------------------------------------------------------
# Aceros NO ALEADOS (B.C.C. ferrítico-perlítico) — el usuario define %C
GRADOS_NO_ALEADOS = {
    "SA-516 Gr 60": {"sy": 220.0, "su": 415.0},
    "SA-516 Gr 70": {"sy": 260.0, "su": 485.0},
    "SA-515 Gr 60": {"sy": 220.0, "su": 400.0},
    "SA-515 Gr 65": {"sy": 240.0, "su": 450.0},
    "SA-515 Gr 70": {"sy": 260.0, "su": 485.0},
}

# Aceros ALEADOS — norma fija, no dependen del slider de %C
GRADOS_ALEADOS = {
    "AISI 4140 (martensítico, temple y revenido)": {
        "sy": 750.0, "su": 950.0, "e_max": 0.16, "n": 0.08,
        "charpy_J": 15.0, "familia": "mart", "pc_nominal": 0.40,
    },
    "AISI 4340 (martensítico, temple y revenido)": {
        "sy": 850.0, "su": 980.0, "e_max": 0.13, "n": 0.07,
        "charpy_J": 12.0, "familia": "mart", "pc_nominal": 0.40,
    },
    "Inoxidable AISI 304 (F.C.C. austenítico)": {
        "sy": 215.0, "su": 505.0, "e_max": 0.55, "n": 0.40,
        "charpy_J": 180.0, "familia": "fcc", "pc_nominal": 0.06,
    },
    "Inoxidable AISI 316 (F.C.C. austenítico)": {
        "sy": 205.0, "su": 515.0, "e_max": 0.52, "n": 0.42,
        "charpy_J": 190.0, "familia": "fcc", "pc_nominal": 0.05,
    },
}


# ----------------------------------------------------------------------
# AJUSTES POR COMPOSICIÓN Y TEMPERATURA
# ----------------------------------------------------------------------
def ajustar_por_carbono_bcc(sy0, su0, pc):
    """No aleados: sy y su suben con %C; la ductilidad y el exponente de
    endurecimiento (n) bajan (menos capacidad de acritud remanente)."""
    sy_c = sy0 + 150.0 * (pc - C_REF)
    su_c = su0 + 300.0 * (pc - C_REF)
    e_max = max(0.35 - 0.40 * (pc - 0.10), 0.03)
    n_hard = 0.25 - 0.10 * ((pc - 0.10) / 0.70)   # 0.25 (bajo %C) -> 0.15 (alto %C)
    return sy_c, su_c, e_max, n_hard


def ajustar_por_temperatura(sy, su, e_max, ttrab, familia):
    """Ablanda a alta T; fragiliza en frío (fuerte en BCC/mart, casi nulo en FCC)."""
    if ttrab >= 20:
        factor_resist = max(0.30, 1 - 0.0012 * (ttrab - 20))
        factor_duct = 1 + 0.0018 * (ttrab - 20)
    else:
        delta = abs(ttrab - 20)
        if familia == "fcc":
            factor_resist = 1 + 0.0004 * delta
            factor_duct = max(0.80, 1 - 0.0004 * delta)
        else:
            factor_resist = 1 + 0.0009 * delta
            factor_duct = max(0.05, 1 - 0.0060 * delta)

    sy_f = sy * factor_resist
    su_f = max(su * factor_resist, sy_f * 1.02)
    # Límite de seguridad visual: el eje Y del gráfico σ-ε está fijo en 0-1000 MPa
    su_f = min(su_f, 980.0)
    sy_f = min(sy_f, su_f * 0.95)
    e_max_f = min(max(e_max * factor_duct, 0.005), 0.58)
    return sy_f, su_f, e_max_f


# ----------------------------------------------------------------------
# CURVA TENSIÓN–DEFORMACIÓN ARMÓNICA (Ludwik/Hollomon + Bridgman)
# ----------------------------------------------------------------------
def generar_curva_tension_deformacion(sy, su, e_max, n_hard, sigma_fract_frac=0.65):
    """
    Devuelve strain, stress_eng, stress_true (mismo tamaño, N_POINTS),
    junto con e_u_eng (deformación de carga máxima) y un flag de si hubo
    estricción o la rotura ocurrió antes de alcanzar la carga máxima.
    La curva termina EXACTAMENTE en e_max (punto de fractura); no se
    dibuja nada más allá.
    """
    e_y = sy / E_MOD
    strain = np.linspace(0.0, e_max, N_POINTS)
    stress_eng = np.zeros(N_POINTS)
    stress_true = np.zeros(N_POINTS)

    # --- Caso 1: fractura frágil pura (rompe dentro de la zona elástica) ---
    if e_max <= e_y:
        stress_eng[:] = E_MOD * strain
        stress_true[:] = stress_eng * (1 + strain)
        return strain, stress_eng, stress_true, e_max, e_y, False

    # --- Parámetros de Ludwik/Hollomon: σ_true = σy + K·εp^n ---
    e_y_true = np.log(1 + e_y)
    e_u_true_total = e_y_true + n_hard              # Considère: εp,u ≈ n
    e_u_eng = np.exp(e_u_true_total) - 1.0           # deformación ingenieril en UTS
    su_true_objetivo = su * np.exp(e_u_true_total)   # σ_true en la carga máxima
    K_hard = (su_true_objetivo - sy) / (n_hard ** n_hard)

    hay_estriccion = e_max > e_u_eng

    # --- Tramo elástico (Hooke) ---
    m_el = strain <= e_y
    stress_eng[m_el] = E_MOD * strain[m_el]
    stress_true[m_el] = stress_eng[m_el] * (1 + strain[m_el])

    if not hay_estriccion:
        # --- Caso 2: rotura dentro de la zona de endurecimiento uniforme ---
        m_pl = strain > e_y
        e_true_tot = np.log(1 + strain[m_pl])
        e_p_true = np.clip(e_true_tot - e_y_true, 0, None)
        s_true = sy + K_hard * (e_p_true ** n_hard)
        stress_true[m_pl] = s_true
        stress_eng[m_pl] = s_true / (1 + strain[m_pl])
        stress_eng = np.minimum(stress_eng, 999.0)
        stress_true = np.minimum(stress_true, 999.0)
        return strain, stress_eng, stress_true, e_max, e_y, False

    # --- Caso 3: curva completa, con estricción hasta la fractura ---
    m_unif = (strain > e_y) & (strain <= e_u_eng)
    e_true_tot_u = np.log(1 + strain[m_unif])
    e_p_true_u = np.clip(e_true_tot_u - e_y_true, 0, None)
    s_true_u = sy + K_hard * (e_p_true_u ** n_hard)
    stress_true[m_unif] = s_true_u
    stress_eng[m_unif] = s_true_u / (1 + strain[m_unif])

    m_neck = strain > e_u_eng
    # Convencional: decaimiento suave (smoothstep, derivada nula en ambos extremos)
    sigma_fractura = su * sigma_fract_frac
    frac = (strain[m_neck] - e_u_eng) / (e_max - e_u_eng + 1e-9)
    frac = np.clip(frac, 0, 1)
    smooth = 3 * frac**2 - 2 * frac**3
    stress_eng[m_neck] = su - (su - sigma_fractura) * smooth

    # Real: continúa el endurecimiento + corrección por triaxialidad (Bridgman aprox.)
    e_true_tot_n = np.log(1 + strain[m_neck])
    e_p_true_n = e_true_tot_n - e_y_true
    s_true_base = sy + K_hard * (e_p_true_n ** n_hard)
    factor_bridgman = 1 + 0.40 * (e_p_true_n - n_hard)
    stress_true[m_neck] = s_true_base * factor_bridgman

    # Límite de seguridad visual: el eje Y del gráfico está fijo en 0-1000 MPa
    stress_eng = np.minimum(stress_eng, 999.0)
    stress_true = np.minimum(stress_true, 999.0)

    return strain, stress_eng, stress_true, e_u_eng, e_y, True


# ----------------------------------------------------------------------
# CURVA CHARPY ARMÓNICA (energía vs temperatura)
# ----------------------------------------------------------------------
def generar_curva_charpy(pc, familia, charpy_ref=None):
    temps = np.linspace(CHARPY_XRANGE[0], CHARPY_XRANGE[1], N_POINTS)

    if familia == "bcc":
        # Desplazamiento fluido: +14°C por cada 0.1% de Carbono
        t_trans = -60.0 + 140.0 * (pc - 0.10)
        ancho = 25.0 + 25.0 * ((pc - 0.10) / 0.70)          # se ensancha con %C
        use = 300.0 - 180.0 * ((pc - 0.10) / 0.70)           # escalón superior baja con %C
        lse = 10.0

        # Se calcula T0 (centro de la sigmoidal) para que E(t_trans) = 20 J exactos
        arg = (2 * TTRANS_REF_J - use - lse) / (use - lse)
        arg = np.clip(arg, -0.999, 0.999)
        t0 = t_trans - ancho * np.arctanh(arg)

        energia = (use + lse) / 2 + (use - lse) / 2 * np.tanh((temps - t0) / ancho)
        return temps, energia, t_trans

    if familia == "fcc":
        plateau = charpy_ref if charpy_ref else 180.0
        low_end = max(plateau - 50.0, 120.0)
        amp = plateau - low_end
        energia = plateau - amp * np.exp(-(temps + 200.0) / 150.0)
        return temps, energia, None

    # martensítico: meseta baja, siempre < 20 J, con leve variación (no perfectamente recta)
    base_j = charpy_ref if charpy_ref else 15.0
    energia = np.clip(base_j + 0.01 * (temps - 20.0), 3.0, 19.0)
    return temps, energia, None


def energia_en_temperatura(temps, energia, ttrab):
    if ttrab < temps[0] or ttrab > temps[-1]:
        return None
    return float(np.interp(ttrab, temps, energia))


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Material")

tipo_acero = st.sidebar.radio(
    "Familia de Acero",
    ["No Aleado (calcular por %C)", "Aleado (norma fija)"],
)

if tipo_acero.startswith("No Aleado"):
    grado_base = st.sidebar.selectbox("Grado base (Norma ASME)", list(GRADOS_NO_ALEADOS.keys()))
    pct_c = st.sidebar.slider("Contenido de Carbono (%C)", 0.10, 0.80, 0.20, 0.01, format="%.2f %%")

    base = GRADOS_NO_ALEADOS[grado_base]
    sy_c, su_c, e_max_c, n_hard = ajustar_por_carbono_bcc(base["sy"], base["su"], pct_c)
    familia = "bcc"
    charpy_ref = None
    nombre_material = f"{grado_base} (%C = {pct_c:.2f}%)"
else:
    grado_aleado = st.sidebar.selectbox("Grado", list(GRADOS_ALEADOS.keys()))
    info = GRADOS_ALEADOS[grado_aleado]
    sy_c, su_c, e_max_c = info["sy"], info["su"], info["e_max"]
    n_hard = info["n"]
    familia = info["familia"]
    charpy_ref = info["charpy_J"]
    pct_c = info["pc_nominal"]
    st.sidebar.caption(f"%C nominal de referencia: {pct_c:.2f}% (fijo, no editable)")
    nombre_material = grado_aleado

ttrab = st.sidebar.number_input(
    "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=500, value=20, step=5
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Los ejes de ambos gráficos están fijos para permitir comparar directamente "
    "distintos materiales y condiciones de temperatura/composición."
)

# ----------------------------------------------------------------------
# CÁLCULOS
# ----------------------------------------------------------------------
sy_final, su_final, e_max_final = ajustar_por_temperatura(sy_c, su_c, e_max_c, ttrab, familia)

strain, stress_eng, stress_true, e_u_eng, e_y, hay_estriccion = generar_curva_tension_deformacion(
    sy_final, su_final, e_max_final, n_hard
)

temps, energia, t_trans = generar_curva_charpy(pct_c, familia, charpy_ref)
energia_ttrab = energia_en_temperatura(temps, energia, ttrab)

# ----------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------
st.title("🔩 Simulador de Comportamiento Mecánico de Materiales")
st.markdown(f"**Material:** {nombre_material} &nbsp;|&nbsp; **Ttrab:** {ttrab} °C")

c1, c2, c3, c4 = st.columns(4)
c1.metric("σy (Límite Elástico)", f"{sy_final:,.0f} MPa")
c2.metric("σu (Resistencia Máxima)", f"{su_final:,.0f} MPa")
c3.metric("Alargamiento a rotura", f"{e_max_final*100:,.1f} %")
c4.metric("Exponente n (Hollomon)", f"{n_hard:.2f}")

st.markdown("---")

# ----------------------------------------------------------------------
# GRÁFICO 1 — TENSIÓN vs DEFORMACIÓN
# ----------------------------------------------------------------------
st.subheader("1️⃣ Curva Tensión – Deformación (σ vs ε)")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=strain, y=stress_eng, mode="lines", name="Curva Convencional (ingenieril)",
    line=dict(color="#1f77b4", width=3, shape="spline", smoothing=0.3),
))
fig1.add_trace(go.Scatter(
    x=strain, y=stress_true, mode="lines", name="Curva Real (verdadera, con corrección de triaxialidad)",
    line=dict(color="#d62728", width=3, dash="dash", shape="spline", smoothing=0.3),
))
if hay_estriccion:
    fig1.add_trace(go.Scatter(
        x=[e_u_eng], y=[su_final], mode="markers+text",
        marker=dict(size=11, color="black", symbol="x"),
        text=["UTS"], textposition="top center", name="Carga Máxima (necking)",
    ))
fig1.add_trace(go.Scatter(
    x=[strain[-1]], y=[stress_eng[-1]], mode="markers+text",
    marker=dict(size=11, color="#7f0000", symbol="star"),
    text=["Fractura"], textposition="bottom right", name="Punto de Fractura",
))
fig1.update_layout(
    xaxis=dict(title="Deformación ε [mm/mm]", range=SIGMA_EPS_XRANGE),
    yaxis=dict(title="Tensión σ [MPa]", range=SIGMA_EPS_YRANGE),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=500, margin=dict(t=30),
)
st.plotly_chart(fig1, use_container_width=True)

with st.expander("📘 Fundamento teórico — Curva σ-ε armónica"):
    st.markdown(
        r"""
**Zona elástica:** Ley de Hooke, $\sigma = E \cdot \varepsilon$, hasta $\sigma_y$.

**Zona plástica uniforme:** ecuación de Ludwik (variante de Hollomon con offset de
fluencia para continuidad perfecta con la zona elástica):

$$\sigma_{real} = \sigma_y + K \cdot \varepsilon_{p}^{\,n}$$

donde $\varepsilon_p$ es la deformación real plástica y $n$ el exponente de
endurecimiento (aceros al carbono: $n \approx 0.15$–$0.25$; inoxidables F.C.C.:
$n \approx 0.40$, mayor capacidad de acritud). El punto de carga máxima (UTS) se
ubica, por el criterio de Considère, donde $\varepsilon_{p} \approx n$.

**Post-estricción:** la tensión convencional decae suavemente (sin tramos
rectos) por la reducción real de sección, mientras que la tensión real sigue
creciendo, corregida por el estado triaxial de tensiones en el cuello
(aproximación tipo Bridgman/Von Mises).

**Punto de fractura:** la curva finaliza exactamente en el alargamiento máximo
calculado; no se grafica nada posterior a la rotura.

> *"Mirar + Conocimiento Técnico = Ver"* — antes de interpretar la curva, la
> **inspección visual** de la probeta (estricción, superficie de fractura)
> confirma si el comportamiento fue dúctil o frágil.
        """
    )

st.markdown("---")

# ----------------------------------------------------------------------
# GRÁFICO 2 — CHARPY
# ----------------------------------------------------------------------
st.subheader("2️⃣ Energía Absorbida vs Temperatura (Ensayo Charpy)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=temps, y=energia, mode="lines", name="Energía absorbida",
    line=dict(color="#2ca02c", width=3, shape="spline", smoothing=0.3),
))
fig2.add_hline(
    y=TTRANS_REF_J, line_dash="dot", line_color="gray",
    annotation_text=f"{TTRANS_REF_J:.0f} J — referencia de Ttrans", annotation_position="bottom right",
)
if t_trans is not None:
    fig2.add_vline(
        x=t_trans, line_dash="dot", line_color="firebrick",
        annotation_text=f"Ttrans ≈ {t_trans:.0f}°C", annotation_position="top",
    )
if energia_ttrab is not None:
    fig2.add_trace(go.Scatter(
        x=[ttrab], y=[energia_ttrab], mode="markers+text",
        marker=dict(size=13, color="#d62728", symbol="circle"),
        text=[f"Ttrab = {ttrab}°C"], textposition="top center", name="Temperatura de Trabajo",
    ))
else:
    st.caption(
        f"⚠️ Ttrab ({ttrab} °C) está fuera del rango típico de ensayo Charpy "
        f"({CHARPY_XRANGE[0]:.0f} a {CHARPY_XRANGE[1]:.0f} °C) y no se marca en el gráfico."
    )

fig2.update_layout(
    xaxis=dict(title="Temperatura [°C]", range=CHARPY_XRANGE),
    yaxis=dict(title="Energía Absorbida [J]", range=CHARPY_YRANGE),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=500, margin=dict(t=30),
)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("📘 Fundamento teórico — Curva Charpy armónica"):
    st.markdown(
        r"""
**B.C.C. (no aleados):** transición dúctil-frágil modelada con una función
tangente hiperbólica de alta resolución. La **Temperatura de Transición**
($T_{trans}$) se define como el punto donde la energía absorbida alcanza
**20 J**. Al aumentar el %C, la curva se desplaza **+14°C por cada 0.1%C** y
se ensancha (transición menos abrupta).

**F.C.C. (inoxidables 304/316):** sin clivaje, energía siempre alta (>120 J),
con una leve pendiente positiva y saturación suave a mayor temperatura — no
existe una transición dúctil-frágil real.

**Martensíticos (AISI 4140/4340):** tenacidad baja y prácticamente constante
(<20 J) en todo el rango de temperatura: son intrínsecamente frágiles.

> *"Mirar + Conocimiento Técnico = Ver"* — la superficie de fractura de la
> probeta Charpy (brillante/cristalina = frágil vs. fibrosa/mate = dúctil) es
> el primer diagnóstico visual, previo a cualquier cálculo.
        """
    )

# ----------------------------------------------------------------------
# DIAGNÓSTICO DE SEGURIDAD ESTRUCTURAL
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("🚨 Diagnóstico de Seguridad Estructural")

alertas_mostradas = False

if familia == "bcc":
    if ttrab < t_trans:
        st.error(
            f"⚠️ **RIESGO DE FRACTURA FRÁGIL POR CLIVAJE** — "
            f"Ttrab ({ttrab} °C) < Ttrans ({t_trans:.0f} °C). "
            f"Energía absorbida estimada: {energia_ttrab:.0f} J." if energia_ttrab is not None
            else f"⚠️ **RIESGO DE FRACTURA FRÁGIL POR CLIVAJE** — Ttrab ({ttrab} °C) < Ttrans ({t_trans:.0f} °C)."
        )
    else:
        st.success(
            f"✅ **COMPORTAMIENTO DÚCTIL SEGURO** — Ttrab ({ttrab} °C) ≥ Ttrans ({t_trans:.0f} °C)."
            + (f" Energía absorbida estimada: {energia_ttrab:.0f} J." if energia_ttrab is not None else "")
        )
    alertas_mostradas = True
elif familia == "fcc":
    st.success(
        "✅ **COMPORTAMIENTO DÚCTIL SEGURO** — Estructura F.C.C. austenítica: sin "
        "transición dúctil-frágil, dúctil incluso a temperaturas criogénicas."
    )
    alertas_mostradas = True
else:  # martensítico
    st.warning(
        "⚠️ **MATERIAL INTRÍNSECAMENTE FRÁGIL** — Tenacidad baja y constante "
        "(<20 J) en todo el rango de temperatura; extremar el control de "
        "defectos y concentradores de tensión."
    )
    alertas_mostradas = True

if ttrab > T_CREEP:
    st.warning(
        f"🔥 **MECANISMO DE DAÑO POR CREEP (FLUENCIA LENTA)** — Ttrab ({ttrab} °C) "
        f"supera los {T_CREEP:.0f} °C: a esta temperatura el material puede sufrir "
        f"deformación progresiva bajo carga sostenida, independientemente del "
        f"resultado del ensayo de impacto."
    )

st.caption(
    "Modelo didáctico simplificado con fines educativos — los valores numéricos "
    "no reemplazan ensayos normalizados (ASTM E8, ASTM E23) ni códigos de diseño "
    "(ASME, API)."
)
