"""
Simulador de Comportamiento Mecánico de Materiales
====================================================
Aplicación educativa en Streamlit para visualizar:
  1. Curva Tensión-Deformación (convencional y real)
  2. Curva de Transición Dúctil-Frágil (Ensayo Charpy)

Basado en apuntes de cátedra de Ingeniería de Materiales / Metalurgia.
Los modelos son simplificaciones didácticas, no valores de norma exactos.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador Mecánico de Materiales",
    page_icon="🔩",
    layout="wide",
)

E_MOD = 210_000.0  # Módulo de Young acero, MPa

# ----------------------------------------------------------------------
# BASES DE DATOS DE MATERIAL
# ----------------------------------------------------------------------
GRADOS_ASME = {
    "SA-516 Gr 60": {"sy": 220.0, "su": 415.0},
    "SA-516 Gr 70": {"sy": 260.0, "su": 485.0},
}

# Multiplicadores según familia cristalográfica / metalúrgica
MATERIALES = {
    "Aceros al Carbono (B.C.C.)": {
        "sy_mult": 1.00, "su_mult": 1.00, "duct_mult": 1.00, "charpy": "bcc",
    },
    "Aceros Inoxidables (F.C.C.)": {
        "sy_mult": 0.90, "su_mult": 0.95, "duct_mult": 1.80, "charpy": "fcc",
    },
    "Alta Resistencia (Martensíticos)": {
        "sy_mult": 1.60, "su_mult": 1.40, "duct_mult": 0.40, "charpy": "mart",
    },
}

C_REF = 0.20  # %C de referencia sobre el cual están dados los valores base del grado ASME


# ----------------------------------------------------------------------
# MODELOS FÍSICOS SIMPLIFICADOS
# ----------------------------------------------------------------------
def ajustar_por_carbono(sy0, su0, pc):
    """Sube sy y su, y baja drásticamente la ductilidad al aumentar %C."""
    sy_c = sy0 + 150.0 * (pc - C_REF)
    su_c = su0 + 300.0 * (pc - C_REF)
    e_max_base = 0.35 - 0.40 * (pc - 0.10)
    e_max_base = max(e_max_base, 0.03)
    return sy_c, su_c, e_max_base


def ajustar_por_temperatura(sy, su, e_max, ttrab, tipo_charpy):
    """Ablanda/fragiliza el material según la temperatura de trabajo."""
    if ttrab >= 20:
        # Ablandamiento térmico: baja resistencia, sube ductilidad
        factor_resist = max(0.30, 1 - 0.0012 * (ttrab - 20))
        factor_duct = 1 + 0.0018 * (ttrab - 20)
    else:
        delta = abs(ttrab - 20)
        if tipo_charpy == "fcc":
            # Los F.C.C. (austeníticos) NO fragilizan en frío
            factor_resist = 1 + 0.0004 * delta
            factor_duct = max(0.75, 1 - 0.0004 * delta)
        else:
            # B.C.C. y martensíticos: fragilización marcada en frío
            factor_resist = 1 + 0.0009 * delta
            factor_duct = max(0.02, 1 - 0.0065 * delta)

    sy_final = sy * factor_resist
    su_final = max(su * factor_resist, sy_final * 1.02)
    e_max_final = max(e_max * factor_duct, 0.002)
    return sy_final, su_final, e_max_final


def generar_curva_tension_deformacion(sy, su, e_max):
    """Genera curva ingenieril (convencional) y curva real hasta carga máxima."""
    e_y = sy / E_MOD

    # Caso frágil puro: casi no hay tramo plástico
    if e_max <= e_y * 1.15:
        strain = np.linspace(0, e_max, 300)
        stress = np.minimum(E_MOD * strain, su)
        e_u = e_max
    else:
        e_u = e_y + 0.55 * (e_max - e_y)  # punto de carga máxima (inicio de estricción)

        # Tramo elástico
        strain_elastica = np.linspace(0, e_y, 60)
        stress_elastica = E_MOD * strain_elastica

        # Tramo plástico de endurecimiento hasta UTS
        strain_plastica = np.linspace(e_y, e_u, 160)[1:]
        frac = (strain_plastica - e_y) / (e_u - e_y)
        stress_plastica = sy + (su - sy) * np.sqrt(frac)

        # Tramo post-UTS: estricción / caída de tensión ingenieril hasta rotura
        strain_estriccion = np.linspace(e_u, e_max, 100)[1:]
        stress_rotura = su * 0.70
        frac2 = (strain_estriccion - e_u) / (e_max - e_u + 1e-9)
        stress_estriccion = su - (su - stress_rotura) * frac2

        strain = np.concatenate([strain_elastica, strain_plastica, strain_estriccion])
        stress = np.concatenate([stress_elastica, stress_plastica, stress_estriccion])

    # Curva real, válida solo hasta el punto de carga máxima (e_u)
    mask_real = strain <= e_u
    strain_real = np.log(1 + strain[mask_real])
    stress_real = stress[mask_real] * (1 + strain[mask_real])

    return strain, stress, strain_real, stress_real, e_u, e_y


def generar_curva_charpy(pc, tipo_charpy):
    """Genera energía absorbida (J) vs temperatura (°C) según familia metalúrgica."""
    temps = np.linspace(-200, 500, 400)

    if tipo_charpy == "bcc":
        t_trans = -50 + 300 * (pc - 0.10)             # se corre a la derecha con %C
        ancho = 20 + 40 * (pc - 0.10) / 0.70           # se suaviza (ensancha) con %C
        use = 150 - 50 * (pc - 0.10) / 0.70            # el escalón superior baja algo
        lse = 10.0
        energia = (use + lse) / 2 + (use - lse) / 2 * np.tanh((temps - t_trans) / ancho)
        return temps, energia, t_trans

    if tipo_charpy == "fcc":
        energia = np.full_like(temps, 180.0) - 0.05 * np.clip(-temps, 0, None)
        return temps, energia, None

    # martensítico: siempre bajo, prácticamente plano
    energia = np.full_like(temps, 15.0) + 0.01 * np.clip(temps, 0, None) * 0
    return temps, energia, None


def energia_en_temperatura(temps, energia, ttrab):
    return float(np.interp(ttrab, temps, energia))


# ----------------------------------------------------------------------
# SIDEBAR — ENTRADAS DEL USUARIO
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Material")

material_tipo = st.sidebar.selectbox(
    "Tipo de Material",
    list(MATERIALES.keys()),
)

pct_c = st.sidebar.slider(
    "Contenido de Carbono (%C)",
    min_value=0.10, max_value=0.80, value=0.20, step=0.01,
    format="%.2f %%",
)

ttrab = st.sidebar.number_input(
    "Temperatura de Trabajo (Ttrab) [°C]",
    min_value=-200, max_value=500, value=20, step=5,
)

grado_base = st.sidebar.radio(
    "Grado base (Referencia ASME)",
    list(GRADOS_ASME.keys()),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Los valores del grado ASME seleccionado se toman como referencia a %C = 0.20 "
    "y luego se ajustan por composición, familia metalúrgica y temperatura de servicio."
)

# ----------------------------------------------------------------------
# CÁLCULOS
# ----------------------------------------------------------------------
mat_info = MATERIALES[material_tipo]
base = GRADOS_ASME[grado_base]

sy_c, su_c, e_max_c = ajustar_por_carbono(base["sy"], base["su"], pct_c)

sy_c *= mat_info["sy_mult"]
su_c *= mat_info["su_mult"]
e_max_c *= mat_info["duct_mult"]
e_max_c = min(max(e_max_c, 0.01), 0.60)

sy_final, su_final, e_max_final = ajustar_por_temperatura(
    sy_c, su_c, e_max_c, ttrab, mat_info["charpy"]
)

strain, stress, strain_real, stress_real, e_u, e_y = generar_curva_tension_deformacion(
    sy_final, su_final, e_max_final
)

temps, energia, t_trans = generar_curva_charpy(pct_c, mat_info["charpy"])
energia_ttrab = energia_en_temperatura(temps, energia, ttrab)

# ----------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------
st.title("🔩 Simulador de Comportamiento Mecánico de Materiales")
st.markdown(
    f"**Material:** {material_tipo} &nbsp;|&nbsp; **%C:** {pct_c:.2f}% &nbsp;|&nbsp; "
    f"**Grado base:** {grado_base} &nbsp;|&nbsp; **Ttrab:** {ttrab} °C"
)

col_metric1, col_metric2, col_metric3 = st.columns(3)
col_metric1.metric("σy (Límite Elástico)", f"{sy_final:,.0f} MPa")
col_metric2.metric("σu (Resistencia Máxima)", f"{su_final:,.0f} MPa")
col_metric3.metric("Alargamiento máx. estimado", f"{e_max_final*100:,.1f} %")

st.markdown("---")

# ----------------------------------------------------------------------
# GRÁFICO 1 — TENSIÓN vs DEFORMACIÓN
# ----------------------------------------------------------------------
st.subheader("1️⃣ Curva Tensión – Deformación (σ vs ε)")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=strain * 100, y=stress, mode="lines", name="Curva Convencional (ingenieril)",
    line=dict(color="#1f77b4", width=3),
))
fig1.add_trace(go.Scatter(
    x=strain_real * 100, y=stress_real, mode="lines", name="Curva Real (verdadera)",
    line=dict(color="#d62728", width=3, dash="dash"),
))
fig1.add_trace(go.Scatter(
    x=[e_u * 100], y=[np.interp(e_u, strain, stress)], mode="markers+text",
    marker=dict(size=10, color="black", symbol="x"),
    text=["σu"], textposition="top center", name="Carga Máxima (UTS)",
))
fig1.update_layout(
    xaxis_title="Deformación ε [%]",
    yaxis_title="Tensión σ [MPa]",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=480,
    margin=dict(t=30),
)
st.plotly_chart(fig1, use_container_width=True)

with st.expander("📘 Fundamento teórico — Curva σ-ε"):
    st.markdown(
        r"""
La **curva convencional (ingenieril)** se calcula con la sección transversal
*original* de la probeta: $\sigma_c = F/A_0$, $\varepsilon = \Delta L/L_0$.

La **curva real** tiene en cuenta la reducción de área durante la deformación
plástica, y es válida solo hasta el punto de carga máxima (inicio de la
estricción), donde deja de ser homogénea:

$$\sigma_r = \sigma_c \cdot (\varepsilon + 1) \qquad \varepsilon_r = \ln(\varepsilon + 1)$$

A mayor **%C**, suben σy y σu pero la ductilidad cae fuertemente (fragilización
por carburos/perlita). A **alta temperatura** (~400 °C) el material se ablanda
y se vuelve más dúctil; a **muy baja temperatura** (~-190 °C) los aceros B.C.C.
y martensíticos rompen de forma frágil, casi sin deformación plástica previa.

> *"Mirar + Conocimiento = Ver"* — antes de interpretar cualquier curva
> tensión-deformación es indispensable la **inspección visual** de la
> probeta ensayada (tipo de fractura, presencia de estricción, superficie de
> rotura) para validar si el comportamiento fue dúctil o frágil.
        """
    )

st.markdown("---")

# ----------------------------------------------------------------------
# GRÁFICO 2 — ENERGÍA ABSORBIDA vs TEMPERATURA (CHARPY)
# ----------------------------------------------------------------------
st.subheader("2️⃣ Energía Absorbida vs Temperatura (Ensayo Charpy)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=temps, y=energia, mode="lines", name="Energía absorbida",
    line=dict(color="#2ca02c", width=3),
))
fig2.add_trace(go.Scatter(
    x=[ttrab], y=[energia_ttrab], mode="markers+text",
    marker=dict(size=13, color="#d62728", symbol="circle"),
    text=[f"Ttrab = {ttrab}°C"], textposition="top center",
    name="Temperatura de Trabajo",
))
if t_trans is not None:
    fig2.add_vline(x=t_trans, line_dash="dot", line_color="gray",
                    annotation_text=f"Ttrans ≈ {t_trans:.0f}°C", annotation_position="top")

fig2.update_layout(
    xaxis_title="Temperatura [°C]",
    yaxis_title="Energía Absorbida [J]",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=480,
    margin=dict(t=30),
)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("📘 Fundamento teórico — Curva Charpy"):
    st.markdown(
        r"""
Los aceros **B.C.C.** presentan una **transición dúctil-frágil** modelada con
una función sigmoidal (tangente hiperbólica), cuyo punto de inflexión es la
**Temperatura de Transición** ($T_{trans}$). Al aumentar el %C, la curva se
desplaza hacia la **derecha** (fragilización a temperaturas más altas) y se
**suaviza** (transición menos abrupta).

Los aceros **F.C.C.** (inoxidables austeníticos) no presentan transición:
mantienen alta energía absorbida incluso a temperaturas criogénicas.

Los aceros **martensíticos** de alta resistencia son intrínsecamente frágiles
en todo el rango de temperaturas (baja energía absorbida constante).

> *"Mirar + Conocimiento = Ver"* — la **inspección visual** de la superficie de
> fractura de la probeta Charpy (brillante/cristalina vs. fibrosa/mate) es el
> primer diagnóstico, previo a cualquier cálculo, del modo de falla dúctil o
> frágil.
        """
    )

# ----------------------------------------------------------------------
# ALERTA DE SEGURIDAD
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("🚨 Diagnóstico de Seguridad Estructural")

if mat_info["charpy"] == "bcc":
    if ttrab < t_trans:
        st.error(
            f"⚠️ **RIESGO DE FALLA FRÁGIL CATASTRÓFICA** — "
            f"Ttrab ({ttrab} °C) < Ttrans ({t_trans:.0f} °C). "
            f"El material se encuentra por debajo de su temperatura de transición: "
            f"la energía absorbida es baja ({energia_ttrab:.0f} J) y la fractura "
            f"esperada es de tipo frágil."
        )
    else:
        st.success(
            f"✅ **COMPORTAMIENTO DÚCTIL SEGURO** — "
            f"Ttrab ({ttrab} °C) ≥ Ttrans ({t_trans:.0f} °C). "
            f"Energía absorbida estimada: {energia_ttrab:.0f} J."
        )
elif mat_info["charpy"] == "fcc":
    st.success(
        f"✅ **COMPORTAMIENTO DÚCTIL SEGURO** — Los aceros inoxidables austeníticos "
        f"(F.C.C.) no presentan transición dúctil-frágil; se mantienen dúctiles "
        f"incluso a temperaturas criogénicas. Energía absorbida estimada: "
        f"{energia_ttrab:.0f} J."
    )
else:  # martensítico
    st.warning(
        f"⚠️ **MATERIAL INTRÍNSECAMENTE FRÁGIL** — Los aceros martensíticos de alta "
        f"resistencia presentan baja tenacidad en todo el rango de temperatura "
        f"(energía absorbida estimada: {energia_ttrab:.0f} J), independientemente "
        f"de Ttrab. Se recomienda extremar el control de defectos y concentradores "
        f"de tensión."
    )

st.caption(
    "Modelo didáctico simplificado con fines educativos — los valores numéricos "
    "no reemplazan ensayos normalizados (ASTM E8, ASTM E23) ni códigos de diseño "
    "(ASME, API)."
)
