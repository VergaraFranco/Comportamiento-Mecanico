"""
Simulador de Comportamiento Mecánico de Materiales — v4
=========================================================
Novedades de esta evolución (sobre la versión anterior):
  - Regla de Compensación C-Mn (ASME Sección II Parte A): por cada 0.01%
    que el %C esté por debajo del máximo de norma, se habilita +0.06% de
    %Mn, sin superar el tope absoluto del grado (1.35% / 1.60% según
    resistencia). Se valida en vivo contra los sliders de %C y %Mn.
  - Eje Y de la curva σ-ε fijado a 0–1000 MPa para TODAS las categorías
    (antes variaba por categoría), priorizando la comparación visual
    directa entre cualquier par de materiales sobre la misma escala.
  - Nueva alerta de seguridad: relación Mn/C < 3 → "Tenacidad Insatisfactoria".
  - Texto de la alerta de clivaje alineado a la cátedra: "CRÍTICO: Riesgo de
    rotura frágil por falta de planos de deslizamiento activos".
  - Renombrado: "Prueba Hidráulica" para el límite del 90% de σy.

Novedades heredadas:
  - Base de datos de Composición Química Teórica (C, Mn, Ni, Cr, Mo, Cu, V)
    extraída de ASME Sección II Parte A / AISI, con cálculo automático del
    Carbono Equivalente: CE = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15.
  - Sliders de %C y %Mn para los aceros no aleados: ambos alimentan la
    Composición Teórica, el CE y la Temperatura de Transición Charpy
    (Ttrans = Ttrans_base + 140·%C − 55·%Mn).
  - Determinación DINÁMICA de fase para aceros no aleados según el
    diagrama Fe-C (A1 = 727°C, A3 = 910 - 230·%C).
  - 4 categorías de material: No Aleados / Alta Resistencia (4140-4340,
    con temperatura de revenido) / Inoxidables / HCP (Zn, Be, Ti-α).
  - Curvas armónicas (Ludwik/Hollomon + Bridgman) sin tramos rectos
    artificiales; exponente n acoplado al factor de ductilidad térmica
    para que el marcador de UTS nunca desaparezca de forma inconsistente.
  - Charpy con tanh de alta resolución (BCC y HCP), saturación suave (FCC)
    y meseta baja (Martensita).
"MIRAR + CONOCIMIENTO TÉCNICO = VER"
"""
import numpy as np
import pandas as pd
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
SIGMA_CLIP = 1980.0                  # tope de seguridad visual, justo debajo del eje más amplio

SIGMA_YRANGE_DEFAULT = [0.0, 1000.0]
SIGMA_YRANGE_ALTA_RESISTENCIA = [0.0, 2000.0]

CHARPY_XRANGE = [-200.0, 300.0]      # °C
CHARPY_YRANGE = [0.0, 350.0]         # J
TTRANS_REF_J = 20.0                   # energía que define la Temperatura de Transición
T_CREEP = 370.0                        # °C, umbral de daño por fluencia lenta
A1_TEMP = 727.0   # °C, línea eutectoide del diagrama Fe-C
TTRANS_BASE_BCC = 0.0   # °C, término independiente de la fórmula de Ttrans para BCC

# ----------------------------------------------------------------------
# BASES DE DATOS — PROPIEDADES MECÁNICAS
# ----------------------------------------------------------------------
GRADOS_NO_ALEADOS = {
    "SA-516 Gr 60": {"sy": 220.0, "su": 415.0, "c_base": 0.24, "mn_base": 0.90,
                      "c_max": 0.24, "mn_nom_max": 1.20, "mn_cap": 1.35},
    "SA-516 Gr 70": {"sy": 260.0, "su": 485.0, "c_base": 0.28, "mn_base": 1.00,
                      "c_max": 0.28, "mn_nom_max": 1.20, "mn_cap": 1.60},
    "SA-515 Gr 60": {"sy": 220.0, "su": 415.0, "c_base": 0.24, "mn_base": 0.70,
                      "c_max": 0.24, "mn_nom_max": 0.90, "mn_cap": 1.35},
    "SA-515 Gr 65": {"sy": 240.0, "su": 450.0, "c_base": 0.26, "mn_base": 0.75,
                      "c_max": 0.26, "mn_nom_max": 0.90, "mn_cap": 1.35},
    "SA-515 Gr 70": {"sy": 260.0, "su": 485.0, "c_base": 0.28, "mn_base": 0.80,
                      "c_max": 0.28, "mn_nom_max": 0.90, "mn_cap": 1.60},
}

GRADOS_ALTA_RESISTENCIA = {
    # Valores de referencia fijos de norma (condición de temple y revenido a
    # temperatura mínima de especificación, la de mayor resistencia): ya no
    # dependen de un control de "Temperatura de Revenido" seleccionable.
    "AISI 4140": {"sy": 1670.0, "su": 1970.0, "e_max": 0.09, "n": 0.06, "charpy_J": 10.0},
    "AISI 4340": {"sy": 1700.0, "su": 2000.0, "e_max": 0.08, "n": 0.055, "charpy_J": 8.0},
}

GRADOS_INOXIDABLES = {
    "AISI 304": {"sy": 215.0, "su": 505.0, "e_max": 0.55, "n": 0.40, "charpy_J": 180.0},
    "AISI 316": {"sy": 205.0, "su": 515.0, "e_max": 0.52, "n": 0.42, "charpy_J": 190.0},
}

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

COMPOSICION_ALTA_RESISTENCIA = {
    "AISI 4140": {"C": 0.40, "Mn": 0.87, "Ni": 0.00, "Cr": 0.95, "Mo": 0.20, "Cu": 0.00, "V": 0.00},
    "AISI 4340": {"C": 0.40, "Mn": 0.70, "Ni": 1.83, "Cr": 0.80, "Mo": 0.25, "Cu": 0.00, "V": 0.00},
}
COMPOSICION_INOXIDABLES = {
    "AISI 304": {"C": 0.06, "Mn": 1.50, "Ni": 9.00, "Cr": 19.00, "Mo": 0.00, "Cu": 0.00, "V": 0.00},
    "AISI 316": {"C": 0.06, "Mn": 1.50, "Ni": 12.00, "Cr": 17.00, "Mo": 2.50, "Cu": 0.00, "V": 0.00},
}
COMPOSICION_HCP = {
    "Zinc (Zn)": {"Elemento base": "Zn", "Pureza teórica": 99.90, "Principales trazas": "Pb, Cd, Fe"},
    "Berilio (Be)": {"Elemento base": "Be", "Pureza teórica": 99.00, "Principales trazas": "BeO, Fe, Al"},
    "Titanio α (Ti-CP)": {"Elemento base": "Ti", "Pureza teórica": 99.20, "Principales trazas": "O, Fe, N"},
}


def calcular_carbono_equivalente(comp: dict) -> float:
    return (
        comp["C"]
        + comp["Mn"] / 6.0
        + (comp["Cr"] + comp["Mo"] + comp["V"]) / 5.0
        + (comp["Ni"] + comp["Cu"]) / 15.0
    )


def calcular_mn_maximo_permitido(grado_data: dict, pct_c: float) -> float:
    deficit_c = max(grado_data["c_max"] - pct_c, 0.0)
    incrementos = deficit_c / 0.01
    extra_mn = incrementos * 0.06
    return min(grado_data["mn_cap"], grado_data["mn_nom_max"] + extra_mn)


def mostrar_bloque_composicion(nombre_material: str, comp: dict, es_acero: bool = True):
    st.markdown("### 🧪 Composición Química Teórica")
    col_tabla, col_ce = st.columns([2, 1])

    if es_acero:
        df_comp = pd.DataFrame(
            {
                "Elemento": ["C", "Mn", "Ni", "Cr", "Mo", "Cu", "V"],
                "% en Peso": [
                    f"{comp['C']:.2f}", f"{comp['Mn']:.2f}", f"{comp['Ni']:.2f}",
                    f"{comp['Cr']:.2f}", f"{comp['Mo']:.2f}", f"{comp['Cu']:.2f}", f"{comp['V']:.2f}",
                ],
            }
        )
        with col_tabla:
            st.table(df_comp.set_index("Elemento"))
        ce = calcular_carbono_equivalente(comp)
        with col_ce:
            st.metric("Carbono Equivalente (CE)", f"{ce:.3f}")
            if ce < 0.40:
                st.success("Buena soldabilidad: bajo riesgo de fisuración en frío, sin precalentamiento crítico.")
            elif ce < 0.60:
                st.warning("Soldabilidad moderada: se recomienda precalentamiento y control de aporte térmico.")
            else:
                st.error("Baja soldabilidad: alto riesgo de fisuración en frío (HAZ). Precalentamiento obligatorio.")
    else:
        df_comp = pd.DataFrame([comp])
        with col_tabla:
            st.table(df_comp.T.rename(columns={0: "Valor"}))
        with col_ce:
            st.info(
                "El Carbono Equivalente (CE) es una fórmula metalúrgica de soldabilidad "
                "válida para aceros Fe-C; no aplica a metales no ferrosos H.C.P."
            )


def determinar_fase_fe_c(ttrab, pc):
    a3 = 910.0 - 230.0 * pc
    if ttrab < A1_TEMP:
        return "bcc", a3
    elif ttrab < a3:
        return "mixta", a3
    else:
        return "fcc", a3


def ajustar_por_carbono_bcc(sy0, su0, pc):
    sy_c = sy0 + 150.0 * (pc - C_REF)
    su_c = su0 + 300.0 * (pc - C_REF)
    e_max = max(0.35 - 0.40 * (pc - 0.10), 0.03)
    n_hard = np.clip(0.25 - 0.10 * ((pc - 0.10) / 0.70), 0.15, 0.25)
    return sy_c, su_c, e_max, n_hard


def ajustar_por_temperatura(sy, su, e_max, ttrab, familia, n_hard=None):
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
    if n_hard is None:
        return sy_f, su_f, e_max_f
    n_f = float(np.clip(n_hard * factor_duct, 0.03, 0.60))
    return sy_f, su_f, e_max_f, n_f






def _maximo_curva_ingenieril(sy, K, n_hard, e_y_true, strain_grid):
    e_true_tot = np.log(1 + strain_grid)
    e_p_true = np.clip(e_true_tot - e_y_true, 0, None)
    s_true = sy + K * (e_p_true ** n_hard)
    s_eng = s_true / (1 + strain_grid)
    idx = int(np.argmax(s_eng))
    return s_eng[idx], strain_grid[idx], s_true[idx]


def resolver_considere_real(sy, su_objetivo, n_hard, e_y_true, e_max_busqueda=0.6, n_grid=3000, iters=40):
    strain_grid = np.linspace(1e-6, e_max_busqueda, n_grid)
    K_lo, K_hi = 1.0, 50_000.0
    for _ in range(iters):
        K_mid = 0.5 * (K_lo + K_hi)
        s_eng_max, _, _ = _maximo_curva_ingenieril(sy, K_mid, n_hard, e_y_true, strain_grid)
        if s_eng_max < su_objetivo:
            K_lo = K_mid
        else:
            K_hi = K_mid
    K_final = 0.5 * (K_lo + K_hi)
    s_eng_max, eps_uts, s_true_uts = _maximo_curva_ingenieril(sy, K_final, n_hard, e_y_true, strain_grid)
    return K_final, eps_uts, s_eng_max, s_true_uts


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
    K_hard, e_u_eng, su_eng_real, su_true_real = resolver_considere_real(
        sy, su, n_hard, e_y_true, e_max_busqueda=max(e_max, 0.6)
    )

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
    smooth = frac**2
    stress_eng[m_neck] = su - (su - sigma_fractura) * smooth

    ep_true_uts = np.log(1 + e_u_eng) - e_y_true
    e_true_tot_n = np.log(1 + strain[m_neck])
    e_p_true_n = e_true_tot_n - e_y_true
    s_true_base = sy + K_hard * (e_p_true_n ** n_hard)
    factor_bridgman = 1 + 0.40 * (e_p_true_n - ep_true_uts)
    stress_true[m_neck] = s_true_base * factor_bridgman

    stress_eng = np.minimum(stress_eng, SIGMA_CLIP)
    stress_true = np.minimum(stress_true, SIGMA_CLIP)
    return strain, stress_eng, stress_true, e_u_eng, e_y, True


def _tanh_con_ttrans_en_20j(temps, t_trans, ancho, use, lse):
    arg = (2 * TTRANS_REF_J - use - lse) / (use - lse)
    arg = np.clip(arg, -0.999, 0.999)
    t0 = t_trans - ancho * np.arctanh(arg)
    return (use + lse) / 2 + (use - lse) / 2 * np.tanh((temps - t0) / ancho)


def generar_curva_charpy(familia, pc=None, pm=0.0, charpy_ref=None, hcp_info=None):
    temps = np.linspace(CHARPY_XRANGE[0], CHARPY_XRANGE[1], N_POINTS)

    if familia == "bcc":
        t_trans = TTRANS_BASE_BCC + 140.0 * pc - 55.0 * pm
        ancho = 25.0 + 25.0 * ((pc - 0.10) / 0.70)
        use = 300.0 - 500.0 * pc + 30.0 * pm
        use = float(np.clip(use, 50.0, 350.0))
        lse = 10.0
        energia = _tanh_con_ttrans_en_20j(temps, t_trans, ancho, use, lse)
        return temps, energia, t_trans

    if familia == "hcp":
        energia = _tanh_con_ttrans_en_20j(
            temps, hcp_info["t_trans"], hcp_info["ancho"], hcp_info["use"], hcp_info["lse"]
        )
        return temps, energia, hcp_info["t_trans"]

    if familia == "fcc":
        plateau = charpy_ref if charpy_ref else 190.0
        low_end = max(plateau - 50.0, 120.0)
        amp = plateau - low_end
        energia = plateau - amp * np.exp(-(temps + 200.0) / 150.0)
        return temps, energia, None

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
    base = GRADOS_NO_ALEADOS[grado_base]

    pct_c = st.sidebar.slider("Contenido de Carbono (%C)", 0.01, 0.80, base["c_base"], 0.01, format="%.2f %%")
    pct_mn = st.sidebar.slider("Contenido de Manganeso (%Mn)", 0.10, 2.00, base["mn_base"], 0.05, format="%.2f %%")
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=1000, value=20, step=5
    )

    mn_max_permitido = calcular_mn_maximo_permitido(base, pct_c)
    if pct_mn > mn_max_permitido:
        st.sidebar.warning(
            f"⚠️ %Mn excede la Regla de Compensación C-Mn para este %C. "
            f"Máximo admisible: {mn_max_permitido:.2f}% (tope de norma: {base['mn_cap']:.2f}%)."
        )
    else:
        st.sidebar.caption(
            f"✓ %Mn dentro de la Regla de Compensación C-Mn (máximo admisible: {mn_max_permitido:.2f}%)."
        )

    sy_c, su_c, e_max_c, n_hard = ajustar_por_carbono_bcc(base["sy"], base["su"], pct_c)
    fase, a3 = determinar_fase_fe_c(ttrab, pct_c)
    nombre_material = f"{grado_base} (%C = {pct_c:.2f}% · %Mn = {pct_mn:.2f}%)"

    comp_actual = {"C": pct_c, "Mn": pct_mn, "Ni": 0.00, "Cr": 0.00, "Mo": 0.00, "Cu": 0.00, "V": 0.00}
    es_acero_actual = True

    if fase == "bcc":
        sy_final, su_final, e_max_final, n_hard = ajustar_por_temperatura(
            sy_c, su_c, e_max_c, ttrab, "bcc", n_hard
        )
        familia_calculo = "bcc"
        charpy_ref = None
    elif fase == "mixta":
        sy_a1, su_a1, e_max_a1, n_a1 = ajustar_por_temperatura(sy_c, su_c, e_max_c, A1_TEMP, "bcc", n_hard)
        sy_aust, su_aust, e_max_aust, n_aust = 40.0, 120.0, 0.55, 0.40
        t_frac = np.clip((ttrab - A1_TEMP) / max(a3 - A1_TEMP, 1e-6), 0, 1)
        sy_final = sy_a1 * (1 - t_frac) + sy_aust * t_frac
        su_final = su_a1 * (1 - t_frac) + su_aust * t_frac
        e_max_final = e_max_a1 * (1 - t_frac) + e_max_aust * t_frac
        n_hard = n_a1 * (1 - t_frac) + n_aust * t_frac
        familia_calculo = "bcc"
        charpy_ref = None
    else:
        sy_aust, su_aust, e_max_aust, n_aust = 40.0, 120.0, 0.55, 0.40
        sy_final, su_final, e_max_final, n_hard = ajustar_por_temperatura(
            sy_aust, su_aust, e_max_aust, ttrab, "fcc", n_aust
        )
        familia_calculo = "fcc"
        charpy_ref = 200.0

    temps, energia, t_trans = generar_curva_charpy(familia_calculo, pc=pct_c, pm=pct_mn, charpy_ref=charpy_ref)

elif categoria == "Aceros Aleados de Alta Resistencia":
    grado_aleado = st.sidebar.selectbox("Grado", list(GRADOS_ALTA_RESISTENCIA.keys()))
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=500, value=20, step=5
    )
    st.sidebar.caption(
        "Valores de σy/σu de referencia según norma (condición de temple y "
        "revenido de máxima resistencia especificada); Ttrab es la condición "
        "de servicio."
    )

    datos_ar = GRADOS_ALTA_RESISTENCIA[grado_aleado]
    sy_final, su_final, e_max_final, n_hard = ajustar_por_temperatura(
        datos_ar["sy"], datos_ar["su"], datos_ar["e_max"], ttrab, "mart", datos_ar["n"]
    )
    fase = "mart"
    nombre_material = f"{grado_aleado} (martensita revenida, referencia de norma)"
    temps, energia, t_trans = generar_curva_charpy("mart", charpy_ref=datos_ar["charpy_J"])

    comp_actual = COMPOSICION_ALTA_RESISTENCIA[grado_aleado]
    es_acero_actual = True

elif categoria == "Aceros Inoxidables":
    grado_inox = st.sidebar.selectbox("Grado", list(GRADOS_INOXIDABLES.keys()))
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=500, value=20, step=5
    )
    info = GRADOS_INOXIDABLES[grado_inox]
    sy_final, su_final, e_max_final, n_hard = ajustar_por_temperatura(
        info["sy"], info["su"], info["e_max"], ttrab, "fcc", info["n"]
    )
    fase = "fcc"
    nombre_material = f"Inoxidable {grado_inox} (F.C.C.)"
    temps, energia, t_trans = generar_curva_charpy("fcc", charpy_ref=info["charpy_J"])

    comp_actual = COMPOSICION_INOXIDABLES[grado_inox]
    es_acero_actual = True

else:
    grado_hcp = st.sidebar.selectbox("Material", list(MATERIALES_HCP.keys()))
    ttrab = st.sidebar.number_input(
        "Temperatura de Trabajo (Ttrab) [°C]", min_value=-200, max_value=500, value=20, step=5
    )
    info = MATERIALES_HCP[grado_hcp]
    sy_final, su_final, e_max_final, n_hard = ajustar_por_temperatura(
        info["sy"], info["su"], info["e_max"], ttrab, "hcp", info["n"]
    )
    fase = "hcp"
    nombre_material = f"{grado_hcp} (H.C.P.)"
    temps, energia, t_trans = generar_curva_charpy("hcp", hcp_info=info)

    comp_actual = COMPOSICION_HCP[grado_hcp]
    es_acero_actual = False

energia_ttrab = energia_en_temperatura(temps, energia, ttrab)

SIGMA_YRANGE = (
    SIGMA_YRANGE_ALTA_RESISTENCIA if categoria == "Aceros Aleados de Alta Resistencia" else SIGMA_YRANGE_DEFAULT
)

st.sidebar.markdown("---")

strain, stress_eng, stress_true, e_u_eng, e_y, hay_estriccion = generar_curva_tension_deformacion(
    sy_final, su_final, e_max_final, n_hard
)

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

st.info(
    "📚 **Fuentes de datos:** Propiedades mecánicas (σy, σu, alargamiento) — "
    "valores mínimos de las tablas de requisitos de tracción de la Norma ASME "
    "Sección II Parte A (especificaciones SA-516, SA-515 y SA-387). "
    "Composición química — límites máximos de la Tabla 1 de las especificaciones "
    "mencionadas en ASME II Parte A. Factores de impacto (desplazamiento de "
    "Ttrans y energía de la meseta) — coeficientes de la tabla de la pág. 23 del "
    "Apunte de Choque 2018 de la cátedra."
)

st.markdown("---")

mostrar_bloque_composicion(nombre_material, comp_actual, es_acero=es_acero_actual)

if categoria == "Aceros No Aleados":
    if pct_mn > mn_max_permitido:
        st.warning(
            f"⚠️ **Regla de Compensación C-Mn (ASME):** con %C = {pct_c:.2f}%, el "
            f"máximo de %Mn admitido por norma es **{mn_max_permitido:.2f}%** "
            f"(tope absoluto del grado: {base['mn_cap']:.2f}%). El valor actual "
            f"(%Mn = {pct_mn:.2f}%) **excede** ese máximo."
        )
    else:
        st.caption(
            f"📐 Regla de Compensación C-Mn (ASME): con %C = {pct_c:.2f}%, el "
            f"máximo de %Mn admitido por norma es {mn_max_permitido:.2f}% "
            f"(tope absoluto del grado: {base['mn_cap']:.2f}%) — cumple."
        )

st.markdown("---")

with st.expander("📈 Dinámica de los Valores Calculados", expanded=False):
    st.markdown(
        """
- **Temperatura de Trabajo (Ttrab):** su aumento provoca un ablandamiento del
  material, reduciendo proporcionalmente σy y σu mientras aumenta el
  alargamiento a rotura (ductilidad).
- **Carbono (%C):** eleva la resistencia mecánica (σy y σu) por el
  endurecimiento de la matriz, pero acorta significativamente el eje de
  deformación (alargamiento) y hunde la energía en Charpy.
- **Manganeso (%Mn):** aumenta el límite elástico y la carga máxima,
  mejorando además la tenacidad al impacto.
- **Metodología de Cálculo:** los valores de la norma (ambiente) se ajustan
  dinámicamente mediante factores de degradación térmica para Ttrab>20°C y
  mediante correlaciones experimentales para la química, permitiendo predecir
  la respuesta del material en condiciones de servicio fuera del estándar de
  laboratorio.
        """
    )

st.markdown("---")
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
    yaxis=dict(title="Tensión σ [MPa]", range=SIGMA_YRANGE),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=500, margin=dict(t=30),
)
st.plotly_chart(fig1, use_container_width=True)
if su_final > SIGMA_YRANGE[1]:
    st.caption(
        f"ℹ️ σu de este material ({su_final:,.0f} MPa) supera el techo fijo del "
        f"gráfico ({SIGMA_YRANGE[1]:,.0f} MPa); la escala se mantiene fija a "
        f"propósito para comparar todos los materiales en los mismos ejes."
    )

with st.expander("📘 Fundamento Teórico Avanzado — Curva Tensión-Deformación (σ-ε)"):
    st.markdown(
        r"""
Este simulador modela el comportamiento de los materiales bajo cargas estáticas
monoaxiales, permitiendo visualizar la transición entre el régimen elástico y
el plástico, así como la diferencia crítica entre los valores ingenieriles y
reales.

**1. El Cerebro Matemático del Modelo**

El gráfico se construye combinando dos leyes fundamentales de la mecánica de
materiales:

- **Zona Elástica:** se rige por la Ley de Hooke ($\sigma = E \cdot \varepsilon$),
  donde la deformación es reversible y proporcional a la carga aplicada.
- **Zona Plástica Uniforme:** se utiliza la Ecuación de Ludwik (variante de
  Hollomon): $\sigma_{real} = \sigma_y + K \cdot \varepsilon_{p}^{\,n}$. El
  modelo calcula dinámicamente el exponente de endurecimiento ($n$) y el
  coeficiente de resistencia ($K$) para que la curva respete los valores de
  UTS y Fluencia de la norma ASME Sección II Parte A.

**2. Curva Convencional vs. Curva Real**

La diferencia visual entre ambas curvas después del punto de carga máxima
(UTS) es el corazón del diagnóstico técnico:

- **Curva Convencional (Azul):** utiliza siempre la sección inicial ($A_0$)
  para el cálculo. Por ello, al iniciarse la estricción (necking), la tensión
  parece caer hasta la rotura.
- **Curva Real (Roja):** considera la reducción instantánea de la sección
  transversal. Por este motivo, la curva real siempre es creciente,
  reflejando el endurecimiento por deformación del material hasta el último
  instante antes de fallar.

**3. Influencia de la Química y la Temperatura**

El simulador ajusta la morfología de las curvas según los parámetros ingresados:

- **Efecto del Carbono (%C):** un incremento en el carbono eleva el límite
  elástico ($\sigma_y$) y la resistencia máxima ($\sigma_u$), pero acorta
  drásticamente el eje horizontal (reducción de ductilidad/alargamiento).
- **Efecto de la Temperatura:** a altas temperaturas (>370°C), el material
  experimenta un ablandamiento que baja la altura de la curva y aumenta su
  extensión, aproximándose al régimen de Creep. Por el contrario, el frío
  extremo fragiliza el acero, eliminando la zona de estricción y forzando una
  fractura brusca.

**⚠️ Alcance y Objetivo Pedagógico**

Esta herramienta tiene un propósito estrictamente didáctico y de apoyo visual
para facilitar la incorporación de saberes de la cátedra.

Es fundamental comprender que las curvas y valores mostrados son modelos
matemáticos simplificados (como la aproximación de Bridgman para el estado
triaxial de tensiones en el cuello) y no representan resultados de ensayos
reales de tracción.

Esta aplicación busca ayudar a "ver" las tendencias metalúrgicas, pero no debe
utilizarse como referencia absoluta para reemplazar los datos de precisión de
la normativa ASME, ASTM E8.
        """
    )

st.markdown("---")

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

with st.expander("📘 Fundamento Teórico Avanzado — Curva Charpy (Energía vs Temperatura)"):
    st.markdown(
        r"""
Es fundamental comprender que la Temperatura de Transición ($T_{trans}$) no es
una propiedad intrínseca del material ni una ley física absoluta, sino un
valor de referencia obtenido mediante correlaciones experimentales. Su valor
depende estrictamente del criterio adoptado; este simulador utiliza el
**Criterio de Energía Absorbida (T4)**, fijando la transición en los **20
Joules**, que es el estándar de seguridad más aceptado para aceros de baja y
media resistencia.

**1. Naturaleza Experimental del Modelo**

El desplazamiento de la curva en el eje de temperaturas se rige por una
ecuación empírica basada en los apuntes de la cátedra:

$$T_{trans} = T_{base} + 140 \cdot \%C - 55 \cdot \%Mn$$

Los coeficientes (14°C por cada 0,1% de C y -5,5°C por cada 0,1% de Mn) son
valores empíricos que representan cómo la química desplaza físicamente el
punto de falla frágil en probetas Charpy con entalla en V.

Asimismo, el Carbono actúa como un "fuertísimo fragilizador" que no solo corre
la transición a mayores temperaturas, sino que "aplasta" la Meseta Superior
($E_{max}$). El simulador modela este efecto reduciendo la energía disponible
incluso en la zona dúctil (de ~320 J a ~50 J) mediante la ecuación:

$$E_{max} = 300 - 500 \cdot \%C + 30 \cdot \%Mn$$

**2. Modelado y Construcción de la Curva**

Para replicar la morfología en "S" observada en laboratorio, se emplea una
función sigmoidal (tangente hiperbólica - tanh). Esta función une de forma
armónica la Meseta Inferior ($E_{min}$) — zona de rotura por clivaje con
energía mínima — con la Meseta Superior ($E_{max}$), centrando la subida en la
temperatura de transición calculada.

**3. El Rol de la Estructura Cristalina**

La morfología de la gráfica responde directamente a la disponibilidad de
sistemas de deslizamiento de la red:

- **B.C.C. (No aleados) y H.C.P. (Zinc, Berilio):** presentan la curva en "S"
  debido a que sus planos de deslizamiento se "congelan" a baja temperatura,
  forzando una falla por clivaje. En materiales HCP la transición es crítica
  al poseer solo 2 sistemas de deslizamiento independientes.
- **F.C.C. (Inoxidables):** poseen 12 sistemas compactos siempre activos,
  manteniendo alta tenacidad incluso en frío extremo; por ello, no presentan
  una transición dúctil-frágil real.
- **Martensita:** debido a su alta distorsión de red, es intrínsecamente
  frágil y se mantiene en una meseta baja (<20 J) constante.

**⚠️ Alcance y Objetivo Pedagógico**

Esta herramienta tiene un propósito estrictamente didáctico. Su función es
servir como un apoyo visual para facilitar la incorporación de saberes y la
comprensión de los fenómenos metalúrgicos analizados en la cátedra.

Es fundamental entender que las gráficas y valores generados son estimaciones
basadas en tendencias generales y modelos experimentales aproximados; por lo
tanto, no representan resultados de ensayos reales de laboratorio ni datos
técnicos de precisión.

Esta aplicación busca ayudar a "ver" el comportamiento de los materiales, pero
no debe utilizarse como referencia absoluta para reemplazar los datos
específicos de la bibliografía oficial, códigos de diseño (ASME/API).
        """
    )

st.markdown("---")
st.subheader("🚨 Diagnóstico de Seguridad Estructural")

if fase in ("bcc", "hcp"):
    if t_trans is not None and ttrab < t_trans:
        st.error(
            f"⚠️ **CRÍTICO: Riesgo de rotura frágil por falta de planos de "
            f"deslizamiento activos** — Ttrab ({ttrab} °C) está por debajo de "
            f"Ttrans ({t_trans:.0f} °C)."
            + (f" Energía absorbida estimada: {energia_ttrab:.0f} J." if energia_ttrab is not None else "")
        )
    else:
        st.success(
            f"✅ **Comportamiento dúctil seguro** — Ttrab ({ttrab} °C) ≥ Ttrans "
            f"({t_trans:.0f} °C)." if t_trans is not None else "✅ **Comportamiento dúctil seguro**."
        )
elif fase == "mixta":
    if t_trans is not None and ttrab < t_trans:
        st.error(
            f"⚠️ **CRÍTICO: Riesgo de rotura frágil por falta de planos de "
            f"deslizamiento activos** — persiste fracción ferrítica (BCC) en la "
            f"mezcla y Ttrab ({ttrab} °C) está por debajo de Ttrans ({t_trans:.0f} °C)."
        )
    else:
        st.success(
            f"✅ **Comportamiento dúctil seguro** — Ttrab ({ttrab} °C) ≥ Ttrans "
            f"({t_trans:.0f} °C), a pesar de la fracción ferrítica remanente."
        )
elif fase == "fcc":
    st.success(
        "✅ **Fisuración rápida no es un problema habitual** — estructura F.C.C. "
        "austenítica, sin transición dúctil-frágil, dúctil incluso a temperaturas "
        "criogénicas."
    )
else:
    st.warning(
        "⚠️ **Material intrínsecamente frágil** — tenacidad baja y constante "
        "(<20 J) en todo el rango de temperatura; extremar el control de "
        "defectos y concentradores de tensión, sin importar Ttrab."
    )

if es_acero_actual and comp_actual.get("C", 0) > 0:
    mn_c_ratio = comp_actual["Mn"] / comp_actual["C"]
    if mn_c_ratio < 3.0:
        st.warning(
            f"⚠️ **Tenacidad Insatisfactoria** — relación Mn/C = {mn_c_ratio:.1f} "
            f"(< 3). El manganeso no alcanza a compensar la fragilización que "
            f"introduce el carbono; se recomienda subir %Mn o bajar %C para "
            f"mejorar la tenacidad al impacto."
        )

if categoria == "Aceros No Aleados" and pct_c > 0.30:
    st.error(
        "⚠️ **Doble Factor de Riesgo:** El elevado contenido de Carbono no solo "
        "desplaza la fragilidad a temperaturas más altas, sino que reduce "
        "drásticamente la capacidad de absorción de energía incluso en la zona "
        "dúctil (caída de la Meseta Superior)."
    )

if ttrab > T_CREEP:
    st.warning(
        f"🔥 **Mecanismo de Daño por Creep (Fluencia Lenta)** — Ttrab ({ttrab} °C) "
        f"supera los {T_CREEP:.0f} °C: el material puede sufrir deformación "
        f"progresiva bajo carga sostenida. Se recomiendan **réplicas metalográficas** "
        f"periódicas in situ para monitorear cavitación y daño por creep, "
        f"independientemente del resultado del ensayo de impacto."
    )

limite_prueba = 0.90 * sy_final
st.info(
    f"🛑 **Prueba Hidráulica** — la tensión aplicada durante cualquier prueba de "
    f"carga no debe superar el **90% de σy** ({limite_prueba:,.0f} MPa) para "
    f"evitar deformación permanente del componente."
)


st.info(
    "Esta herramienta es un apoyo visual didáctico para facilitar la "
    "incorporación de saberes. Los valores son estimaciones basadas en modelos "
    "matemáticos y tendencias generales; no representan ensayos reales de "
    "laboratorio y no deben usarse como referencia absoluta para diseño."
)

st.markdown("---")

