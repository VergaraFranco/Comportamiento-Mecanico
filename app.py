"""
Simulador de Comportamiento Mecánico de Materiales — v3
=========================================================
Novedades v3:
  - Determinación DINÁMICA de fase para aceros no aleados según el
    diagrama Fe-C (A1 = 727°C, A3 = 910 - 230·%C).
  - 4 categorías de material: No Aleados / Alta Resistencia (4140-4340,
    con temperatura de revenido) / Inoxidables / HCP (Zn, Be, Ti-α).
  - Curvas armónicas (Ludwik/Hollomon + Bridgman) sin tramos rectos
    artificiales; Charpy con tanh de alta resolución (BCC y HCP),
    saturación suave (FCC) y meseta baja (Martensita).

"MIRAR + CONOCIMIENTO TÉCNICO = VER"
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(page_title="Simulador Mecánico de Materiales", page_icon="🔩", layout="wide")

E_MOD = 210_000.0     # Módulo de Young, MPa (aproximado, aceros y aleaciones metálicas)
C_REF = 0.20          # %C de referencia de los grados ASME base
N_POINTS = 500        # resolución fina para curvas armónicas

# Ejes fijos para comparación visual entre materiales/condiciones
SIGMA_EPS_XRANGE = [0.0, 0.6]       # deformación
# SIGMA_EPS_YRANGE = [0.0, 2000.0]    # MPa (ampliado para cubrir martensita revenida hasta ~1970 MPa)
SIGMA_CLIP = 1980.0                  # tope de seguridad visual, justo debajo del eje
CHARPY_XRANGE = [-200.0, 300.0]      # °C
CHARPY_YRANGE = [0.0, 350.0]         # J
TTRANS_REF_J = 20.0                   # energía que define la Temperatura de Transición
T_CREEP = 370.0                        # °C, umbral de daño por fluencia lenta

A1_TEMP = 727.0   # °C, línea eutectoide del diagrama Fe-C

# ----------------------------------------------------------------------
# BASES DE DATOS
# ----------------------------------------------------------------------
GRADOS_NO_ALEADOS = {
    "SA-516 Gr 60": {"sy": 220.0, "su": 415.0},
    "SA-516 Gr 70": {"sy": 260.0, "su": 485.0},
    "SA-515 Gr 60": {"sy": 220.0, "su": 400.0},
    "SA-515 Gr 65": {"sy": 240.0, "su": 450.0},
    "SA-515 Gr 70": {"sy": 260.0, "su": 485.0},
}

# AISI 4140 / 4340 — martensita revenida, resistencia depende de T° de revenido
GRADOS_ALTA_RESISTENCIA = {
    "AISI 4140": {"su_max": 1970.0, "su_min": 700.0, "charpy_min": 8.0, "charpy_max": 16.0},
    "AISI 4340": {"su_max": 2000.0, "su_min": 750.0, "charpy_min": 7.0, "charpy_max": 14.0},
}
T_REVENIDO_MIN, T_REVENIDO_MAX = 200.0, 650.0

# Inoxidables austeníticos — F.C.C., sin transición
GRADOS_INOXIDABLES = {
    "AISI 304": {"sy": 215.0, "su": 505.0, "e_max": 0.55, "n": 0.40, "charpy_J": 180.0},
    "AISI 316": {"sy": 205.0, "su": 515.0, "e_max": 0.52, "n": 0.42, "charpy_J": 190.0},
}

# Materiales H.C.P. — solo 2 sistemas de deslizamiento -> transición dúctil-frágil propia
MATERIALES_HCP = {
    "Zinc (Zn)": {"sy": 60.0, "su": 150.0, "e_max": 0.35, "n": 0.25,
                  "use": 70.0, "lse": 5.0, "t_trans": 10.0, "ancho": 30.0},
    "Berilio (Be)": {"sy": 240.0, "su": 370.0, "e_max": 0.03, "n": 0.05,
                     "use": 25.0, "lse": 2.0, "t_trans": 250.0, "ancho": 80.0},
    "Titanio α (Ti-CP)": {"sy": 300.0, "su": 450.0, "e_max": 0.22, "n": 0.15,
                          "use": 110.0, "lse": 8.0, "t_trans": -80.0, "ancho": 40.0},
}

ESTRUCTURA_TAG = {
    "bcc": "BCC", "mixta": "BCC + FCC", "fcc": "FCC", "mart": "MARTENSITA", "hcp": "HCP",
}
ESTRUCTURA_DESC = {
    "bcc": "Ferrita + Perlita (BCC)",
    "mixta": "Austenita + Ferrita (Mezcla)",
    "fcc": "Austenita (FCC)",
    "mart": "Martensita Revenida",
    "hcp": "Hexagonal Compacta (HCP)",
}


# ----------------------------------------------------------------------
# DIAGRAMA Fe-C — DETERMINACIÓN DINÁMICA DE FASE (no aleados)
# ----------------------------------------------------------------------
def determinar_fase_fe_c(ttrab, pc):
    """Devuelve (fase, A3) según A1=727°C y A3=910-230·%C."""
    a3 = 910.0 - 230.0 * pc
    if ttrab < A1_TEMP:
        return "bcc", a3
    elif ttrab < a3:
        return "mixta", a3
    else:
        return "fcc", a3


# ----------------------------------------------------------------------
# AJUSTES POR COMPOSICIÓN Y TEMPERATURA
# ----------------------------------------------------------------------
def ajustar_por_carbono_bcc(sy0, su0, pc):
    sy_c = sy0 + 150.0 * (pc - C_REF)
    su_c = su0 + 300.0 * (pc - C_REF)
    e_max = max(0.35 - 0.40 * (pc - 0.10), 0.03)
    n_hard = np.clip(0.25 - 0.10 * ((pc - 0.10) / 0.70), 0.15, 0.25)
    return sy_c, su_c, e_max, n_hard


def ajustar_por_temperatura(sy, su, e_max, ttrab, familia):
    """Ablanda a alta T; fragiliza en frío (fuerte en BCC/HCP/mart, casi nulo en FCC)."""
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
    su_f = min(su_f, SIGMA_CLIP)
    sy_f = min(sy_f, su_f * 0.95)
    e_max_f = min(max(e_max * factor_duct, 0.005), 0.58)
    return sy_f, su_f, e_max_f


def propiedades_martensita_revenida(t_revenido, grado):
    """AISI 4140 / 4340: la resistencia cae y la ductilidad sube con la
    temperatura de revenido (parámetro del tratamiento térmico, distinto
    de la temperatura de servicio Ttrab)."""
    datos = GRADOS_ALTA_RESISTENCIA[grado]
    frac = np.clip((t_revenido - T_REVENIDO_MIN) / (T_REVENIDO_MAX - T_REVENIDO_MIN), 0, 1)
    su = datos["su_max"] - (datos["su_max"] - datos["su_min"]) * frac
    sy = 0.85 * su
    e_max = 0.05 + 0.20 * frac
    n_hard = 0.05 + 0.07 * frac
    charpy_j = datos["charpy_min"] + (datos["charpy_max"] - datos["charpy_min"]) * frac
    return sy, su, e_max, n_hard, charpy_j


# ----------------------------------------------------------------------
# CURVA TENSIÓN–DEFORMACIÓN ARMÓNICA (Ludwik/Hollomon + Bridgman)
# ----------------------------------------------------------------------
def generar_curva_tension_deformacion(sy, su, e_max, n_hard, sigma_fract_frac=0.65):
    e_y = sy / E_MOD
    strain = np.linspace(0.0, e_max, N_POINTS)
    stress_eng = np.zeros(N_POINTS)
    stress_true = np.zeros(N_POINTS)

    if e_max <= e_y:
        stress_eng[:] = E_MOD * strain
        stress_true[:] = stress_eng * (1 + strain)
        return strain, stress_eng, stress_true, e_max, e_y, False

    e_y_true = np.log(1 + e_y)
    e_u_true_total = e_y_true + n_hard
    e_u_eng = np.exp(e_u_true_total) - 1.0
    su_true_objetivo = su * np.exp(e_u_true_total)
    K_hard = (su_true_objetivo - sy) / (n_hard ** n_hard)

    hay_estriccion = e_max > e_u_eng

    m_el = strain <= e_y
    stress_eng[m_el] = E_MOD * strain[m_el]
    stress_true[m_el] = stress_eng[m_el] * (1 + strain[m_el])

    if not hay_estriccion:
        m_pl = strain > e_y
        e_true_tot = np.log(1 + strain[m_pl])
        e_p_true = np.clip(e_true_tot - e_y_true, 0, None)
        s_true = sy + K_hard * (e_p_true ** n_hard)
        stress_true[m_pl] = s_true
        stress_eng[m_pl] = s_true / (1 + strain[m_pl])
        stress_eng = np.minimum(stress_eng, SIGMA_CLIP)
        stress_true = np.minimum(stress_true, SIGMA_CLIP)
        return strain, stress_eng, stress_true, e_max, e_y, False

    m_unif = (strain > e_y) & (strain <= e_u_eng)
    e_true_tot_u = np.log(1 + strain[m_unif])
    e_p_true_u = np.clip(e_true_tot_u - e_y_true, 0, None)
    s_true_u = sy + K_hard * (e_p_true_u ** n_hard)
    stress_true[m_unif] = s_true_u
    stress_eng[m_unif] = s_true_u / (1 + strain[m_unif])

    m_neck = strain > e_u_eng
    sigma_fractura = su * sigma_fract_frac
    frac = (strain[m_neck] - e_u_eng) / (e_max - e_u_eng + 1e-9)
    frac = np.clip(frac, 0, 1)
    smooth = 3 * frac**2 - 2 * frac**3
    stress_eng[m_neck] = su - (su - sigma_fractura) * smooth

    e_true_tot_n = np.log(1 + strain[m_neck])
    e_p_true_n = e_true_tot_n - e_y_true
    s_true_base = sy + K_hard * (e_p_true_n ** n_hard)
    factor_bridgman = 1 + 0.40 * (e_p_true_n - n_hard)
    stress_true[m_neck] = s_true_base * factor_bridgman

    stress_eng = np.minimum(stress_eng, SIGMA_CLIP)
    stress_true = np.minimum(stress_true, SIGMA_CLIP)

    return strain, stress_eng, stress_true, e_u_eng, e_y, True


# ----------------------------------------------------------------------
# CURVA CHARPY ARMÓNICA (energía vs temperatura)
# ----------------------------------------------------------------------
def _tanh_con_ttrans_en_20j(temps, t_trans, ancho, use, lse):
    """tanh cuyo T0 se resuelve para que E(t_trans) = 20 J exactos."""
    arg = (2 * TTRANS_REF_J - use - lse) / (use - lse)
    arg = np.clip(arg, -0.999, 0.999)
    t0 = t_trans - ancho * np.arctanh(arg)
    return (use + lse) / 2 + (use - lse) / 2 * np.tanh((temps - t0) / ancho)


def generar_curva_charpy(familia, pc=None, charpy_ref=None, hcp_info=None):
    temps = np.linspace(CHARPY_XRANGE[0], CHARPY_XRANGE[1], N_POINTS)

    if familia == "bcc":
        t_trans = -60.0 + 140.0 * (pc - 0.10)               # +14°C cada 0.1%C
        ancho = 25.0 + 25.0 * ((pc - 0.10) / 0.70)
        use = 300.0 - 180.0 * ((pc - 0.10) / 0.70)
        lse = 10.0
        energia = _tanh_con_ttrans_en_20j(temps, t_trans, ancho, use, lse)
        return temps, energia, t_trans

    if familia == "hcp":
        energia = _tanh_con_ttrans_en_20j(
            temps, hcp_info["t_trans"], hcp_info["ancho"], hcp_info["use"], hcp_info["lse"]
        )
        return temps, energia, hcp_info["t_trans"]

    if familia == "fcc":
        plateau = charpy_ref if charpy_ref else 180.0
        low_end = max(plateau - 50.0, 120.0)
        amp = plateau - low_end
        energia = plateau - amp * np.exp(-(temps + 200.0) / 150.0)
        return temps, energia, None

    # martensítico: meseta baja, siempre < 20 J
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

categoria = st.sidebar.selectbox(
    "Categoría de Material",
    ["Aceros No Aleados", "Aceros Aleados de Alta Resistencia", "Aceros Inoxidables", "Materiales H.C.P."],
)

if categoria == "Aceros No Aleados":
    grado_base = st.sidebar.selectbox("Grado base (Norma ASME)", list(GRADOS_NO_ALEADOS.keys()))
    pct_c = st.sidebar.slider("Contenido de Carbono (%C)", 0.01, 0.80, 0.20, 0.01, format="%.2f %%")
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=1000, value=20, step=5
    )

    base = GRADOS_NO_ALEADOS[grado_base]
    sy_c, su_c, e_max_c, n_hard = ajustar_por_carbono_bcc(base["sy"], base["su"], pct_c)

    fase, a3 = determinar_fase_fe_c(ttrab, pct_c)
    nombre_material = f"{grado_base} (%C = {pct_c:.2f}%)"

    if fase == "bcc":
        sy_final, su_final, e_max_final = ajustar_por_temperatura(sy_c, su_c, e_max_c, ttrab, "bcc")
        familia_calculo = "bcc"
        charpy_ref = None
    elif fase == "mixta":
        # Interpola entre la ferrita (en A1) y una austenita blanda de referencia (en A3)
        sy_a1, su_a1, e_max_a1 = ajustar_por_temperatura(sy_c, su_c, e_max_c, A1_TEMP, "bcc")
        sy_aust, su_aust, e_max_aust = 40.0, 120.0, 0.55
        t_frac = np.clip((ttrab - A1_TEMP) / max(a3 - A1_TEMP, 1e-6), 0, 1)
        sy_final = sy_a1 * (1 - t_frac) + sy_aust * t_frac
        su_final = su_a1 * (1 - t_frac) + su_aust * t_frac
        e_max_final = e_max_a1 * (1 - t_frac) + e_max_aust * t_frac
        familia_calculo = "bcc"   # conserva fracción ferrítica -> riesgo de clivaje remanente
        charpy_ref = None
    else:  # fase == "fcc" (austenita ya formada a esa Ttrab)
        sy_aust, su_aust, e_max_aust = 40.0, 120.0, 0.55
        sy_final, su_final, e_max_final = ajustar_por_temperatura(sy_aust, su_aust, e_max_aust, ttrab, "fcc")
        familia_calculo = "fcc"
        charpy_ref = 150.0

    temps, energia, t_trans = generar_curva_charpy(familia_calculo, pc=pct_c, charpy_ref=charpy_ref)

elif categoria == "Aceros Aleados de Alta Resistencia":
    grado_aleado = st.sidebar.selectbox("Grado", list(GRADOS_ALTA_RESISTENCIA.keys()))
    t_revenido = st.sidebar.slider(
        "Temperatura de Revenido [°C]", int(T_REVENIDO_MIN), int(T_REVENIDO_MAX), 350, 10
    )
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=500, value=20, step=5
    )
    st.sidebar.caption(
        "La T° de revenido es un parámetro del tratamiento térmico (fija la "
        "resistencia); Ttrab es la condición de servicio."
    )

    sy_ht, su_ht, e_max_ht, n_hard, charpy_ref = propiedades_martensita_revenida(t_revenido, grado_aleado)
    sy_final, su_final, e_max_final = ajustar_por_temperatura(sy_ht, su_ht, e_max_ht, ttrab, "mart")
    fase = "mart"
    nombre_material = f"{grado_aleado} (revenido a {t_revenido}°C)"
    temps, energia, t_trans = generar_curva_charpy("mart", charpy_ref=charpy_ref)

elif categoria == "Aceros Inoxidables":
    grado_inox = st.sidebar.selectbox("Grado", list(GRADOS_INOXIDABLES.keys()))
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=500, value=20, step=5
    )
    info = GRADOS_INOXIDABLES[grado_inox]
    n_hard = info["n"]
    sy_final, su_final, e_max_final = ajustar_por_temperatura(info["sy"], info["su"], info["e_max"], ttrab, "fcc")
    fase = "fcc"
    nombre_material = f"Inoxidable {grado_inox} (F.C.C.)"
    temps, energia, t_trans = generar_curva_charpy("fcc", charpy_ref=info["charpy_J"])

else:  # Materiales H.C.P.
    grado_hcp = st.sidebar.selectbox("Material", list(MATERIALES_HCP.keys()))
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=500, value=20, step=5
    )
    info = MATERIALES_HCP[grado_hcp]
    n_hard = info["n"]
    sy_final, su_final, e_max_final = ajustar_por_temperatura(info["sy"], info["su"], info["e_max"], ttrab, "hcp")
    fase = "hcp"
    nombre_material = f"{grado_hcp} (H.C.P.)"
    temps, energia, t_trans = generar_curva_charpy("hcp", hcp_info=info)

energia_ttrab = energia_en_temperatura(temps, energia, ttrab)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Los ejes de ambos gráficos están fijos para permitir comparar directamente "
    "distintos materiales y condiciones."
)

# ----------------------------------------------------------------------
# CURVA σ-ε
# ----------------------------------------------------------------------
strain, stress_eng, stress_true, e_u_eng, e_y, hay_estriccion = generar_curva_tension_deformacion(
    sy_final, su_final, e_max_final, n_hard
)

# ----------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------
st.title("🔩 Simulador de Comportamiento Mecánico de Materiales")
st.markdown(f"**Material:** {nombre_material} &nbsp;|&nbsp; **Ttrab:** {ttrab} °C")

st.markdown(
    f"""
    <div style="padding:14px 18px;border-radius:10px;background-color:#0e2f44;
                border:1px solid #1f77b4;margin-bottom:8px;">
        <span style="font-size:1.05rem;color:#cfe8ff;">Fase Metalúrgica Identificada</span><br>
        <span style="font-size:1.9rem;font-weight:700;color:#ffffff;">{ESTRUCTURA_TAG[fase]}</span>
        &nbsp;—&nbsp;<span style="font-size:1.1rem;color:#e6f2ff;">{ESTRUCTURA_DESC[fase]}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("σy (Límite Elástico)", f"{sy_final:,.0f} MPa")
c2.metric("σu (Resistencia Máxima)", f"{su_final:,.0f} MPa")
c3.metric("Alargamiento a rotura", f"{e_max_final*100:,.1f} %")

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

**Zona plástica uniforme:** ecuación de Ludwik (variante de Hollomon con offset
de fluencia): $\sigma_{real} = \sigma_y + K \cdot \varepsilon_{p}^{\,n}$. El punto
de carga máxima (UTS) se ubica, por el criterio de Considère, donde
$\varepsilon_{p} \approx n$.

**Post-estricción:** la tensión convencional decae suavemente por la reducción
real de sección; la tensión real sigue creciendo, corregida por el estado
triaxial de tensiones en el cuello (aproximación tipo Bridgman/Von Mises).

**Aceros al Carbono (no aleados):** el %C sube σy y σu, pero reduce fuertemente
la ductilidad. **AISI 4140/4340:** la resistencia (hasta ~1970-2000 MPa) depende
de la temperatura de revenido del tratamiento térmico, no de Ttrab.

> *"Mirar + Conocimiento Técnico = Ver"* — la **inspección visual** de la
> probeta (estricción, superficie de fractura) confirma si el comportamiento
> fue dúctil o frágil.
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
**B.C.C. (no aleados) y H.C.P. (Zn, Be, Ti-α):** ambas presentan transición
dúctil-frágil (tanh de alta resolución). La **Temperatura de Transición**
($T_{trans}$) se define donde la energía absorbida alcanza **20 J**. En los
no aleados, el %C desplaza la curva **+14°C por cada 0.1%C**. Los H.C.P.
transicionan por tener solo **2 sistemas de deslizamiento** disponibles.

**F.C.C. (inoxidables 304/316):** sin clivaje, energía siempre alta (>120 J),
con saturación suave — no existe transición dúctil-frágil real.

**Martensíticos (AISI 4140/4340):** tenacidad baja y prácticamente constante
(<20 J): son intrínsecamente frágiles, independientemente de Ttrab.

> *"Mirar + Conocimiento Técnico = Ver"* — la superficie de fractura de la
> probeta Charpy (brillante/cristalina = frágil vs. fibrosa/mate = dúctil) es
> el primer diagnóstico visual, previo a cualquier cálculo.
        """
    )

# ----------------------------------------------------------------------
# FUNDAMENTOS DE SEGURIDAD
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("🚨 Fundamentos de Seguridad")

if fase in ("bcc", "mixta", "hcp"):
    if t_trans is not None and ttrab < t_trans:
        st.error(
            f"⚠️ **Peligro de Rotura por Clivaje** — Ttrab ({ttrab} °C) está por "
            f"debajo de Ttrans ({t_trans:.0f} °C)."
            + (f" Energía absorbida estimada: {energia_ttrab:.0f} J." if energia_ttrab is not None else "")
        )
    else:
        st.success(
            f"✅ **Comportamiento dúctil seguro** — Ttrab ({ttrab} °C) ≥ Ttrans "
            f"({t_trans:.0f} °C)." if t_trans is not None else "✅ **Comportamiento dúctil seguro**."
        )
elif fase == "fcc":
    st.success(
        "✅ **Fisuración rápida no es un problema habitual** — estructura F.C.C. "
        "austenítica, sin transición dúctil-frágil, dúctil incluso a temperaturas "
        "criogénicas."
    )
else:  # martensita
    st.warning(
        "⚠️ **Material intrínsecamente frágil** — tenacidad baja y constante "
        "(<20 J) en todo el rango de temperatura; extremar el control de "
        "defectos y concentradores de tensión, sin importar Ttrab."
    )

if ttrab > T_CREEP:
    st.warning(
        f"🔥 **Mecanismo de Daño por Creep (Fluencia Lenta)** — Ttrab ({ttrab} °C) "
        f"supera los {T_CREEP:.0f} °C: el material puede sufrir deformación "
        f"progresiva bajo carga sostenida, independientemente del ensayo de impacto."
    )

st.caption(
    "Modelo didáctico simplificado con fines educativos — los valores numéricos "
    "no reemplazan ensayos normalizados (ASTM E8, ASTM E23) ni códigos de diseño "
    "(ASME, API)."
)
