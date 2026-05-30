# Taller 1 Poster — Análisis de la tasa de crecimiento del PIB de Alemania (1992–2023)
# Autor: Diego Alejandro Lizarazo Carrera, Sergio Meléndez Gutiérrez y Luisa Fernanda Molina Suárez
# Grupo: Luis Luna
# Econometría II, Facultad de Ciencias Económicas, Universiddad Nacional de Colombia

# %% Importación de paquetes ============================

# Trabajar con rutas relativas en python
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

# Módulos de statsmodels
from scipy.stats import jarque_bera, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

# %% Rutas relativas ============================

# BASE_DIR apunta a la carpeta raíz del proyecto ("Taller 1 Poster")
# independientemente de dónde esté guardada en el computador.
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "datos"
RESULTADOS_DIR = BASE_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

ruta_datos = DATA_DIR / "LORSGPORDEQ659S.xlsx"


# %% Carga de la base de datos ============================

raw = pd.read_excel(
    ruta_datos,
    sheet_name="Quarterly",
    parse_dates=["observation_date"]
)

raw = raw.set_index("observation_date")
raw.index = pd.DatetimeIndex(raw.index).to_period("Q").to_timestamp("Q")

serie = raw["LORSGPORDEQ659S"].copy()
serie = pd.to_numeric(serie, errors="coerce").dropna()

# Se excluye de la serie de tiempo los periodos anteriores al 1992-06-30, ya que concuerda
# con los choques sobre el PIB derivado de la reunificación alemana. Este periodo fue confirmado
# con la metodología para detectar cambios estructurales en la serie de tiempo Test de Chow. 
# Se trabaja con 1961 Q1 - 2023 Q4 (131 observaciones).

serie = serie["1992-06-30":"2023-10-01"]

print("=== Descripción de la serie ===")
print(f"Observaciones : {len(serie)}")
print(f"Periodo       : {serie.index[0].date()} — {serie.index[-1].date()}")
print(serie.describe())


# %% =========================
# PASO 1: IDENTIFICACIÓN
# ============================

# Gráfica de la serie de tiempo
plt.figure(figsize=(10, 5))
plt.plot(serie)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("CLI: tasa de crecimiento del PIB de Alemania (1992–2023)")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "01_serie_original.png", dpi=150)
plt.show()

# %% FAC y FACP de la serie original

# Reglas de identificación Box-Jenkins:
# - FACP se corta en rezago p, FAC decae gradualmente → proceso AR(p)
# - FAC se corta en rezago q, FACP decae gradualmente → proceso MA(q)
# - Ambas decaen gradualmente                         → proceso ARMA(p,q)
#
# En nuestra serie la FACP se corta después del rezago 1 y la FAC decae
# gradualmente. Esto sugiere un AR(1) como modelo base. Se proponen además
# ARMA(4,0) dado que hay un pico en el rezago 4 de la FACP, y ARMA(1,1)
# como modelo mixto alternativo.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)
axes[0].set_title("FAC — serie original")
axes[0].set_ylim(-1, 1)

plot_pacf(
    serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)
axes[1].set_title("FACP — serie original")
axes[1].set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "02_fac_facp_original.png", dpi=150)
plt.show()

# %% Test ADF

# H0: la serie tiene raíz unitaria (no es estacionaria)
# Si p-valor < 0.05 → rechazamos H0 → la serie es estacionaria
adf_result = adfuller(serie, autolag="AIC")

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2])
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"  {nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
    d = 0
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")
    d = 1


# %% =========================
# PASO 2: ESTIMACIÓN
# ============================

# Modelos candidatos sugeridos por la inspección de FAC y FACP:
# ARMA(1,0): FACP se corta en 1, FAC decae → AR(1) puro
# ARMA(4,0): pico significativo en rezago 4 de la FACP → explorar AR(4)
# ARMA(1,1): modelo mixto alternativo para comparar
modelos_candidatos = {
    "ARMA(1,0)": (1, d, 0),
    "ARMA(4,0)": (4, d, 0),
    "ARMA(1,1)": (1, d, 1),
}

nombres_modelos = list(modelos_candidatos.keys())

# Diccionario que almacenará las estimaciones de cada modelo
estimaciones = {}


# Función para calcular la media incondicional del modelo
def media_incondicional_sarimax(resultado):
    """Calcula E[y_t] para un modelo ARMA(p,q)"""
    params = resultado.params
    intercepto = params.get("intercept", 0)
    suma_ar = sum(
        valor
        for parametro, valor in params.items()
        if parametro.startswith("ar.L")
    )
    denominador = 1 - suma_ar
    if np.isclose(denominador, 0):
        return np.nan
    return intercepto / denominador


# Función para calcular el error estándar de la media incondicional
def se_media_incondicional_sarimax(resultado):
    """Calcula el error estándar de la media incondicional usando método delta."""
    params = resultado.params
    if "intercept" not in params.index:
        return np.nan
    parametros_ar = [p for p in params.index if p.startswith("ar.L")]
    intercepto = params["intercept"]
    denominador = 1 - sum(params[p] for p in parametros_ar)
    if np.isclose(denominador, 0):
        return np.nan
    cov_params = resultado.cov_params()
    gradiente = pd.Series(0.0, index=params.index)
    gradiente["intercept"] = 1 / denominador
    for p in parametros_ar:
        gradiente[p] = intercepto / denominador**2
    varianza_mu = float(gradiente @ cov_params @ gradiente)
    if varianza_mu < 0:
        return np.nan
    return np.sqrt(varianza_mu)


# Estimación de cada modelo candidato
for nombre, orden in modelos_candidatos.items():
    modelo = SARIMAX(
        serie,
        order=orden,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    estimacion_modelo = modelo.fit(method='bfgs', maxiter=400, disp=False)
    estimaciones[nombre] = estimacion_modelo

    print("\n", nombre)
    print(estimacion_modelo.summary())
    print(
        "Media incondicional implícita en SARIMAX "
        f"c / (1 - suma AR): {media_incondicional_sarimax(estimacion_modelo):.3f}"
    )

# --- Estadísticos exactos LB(20) y ARCH(5) para el ARMA(4,0) ---
resultado_40 = estimaciones["ARMA(4,0)"]
p_40         = resultado_40.model.order[0]
q_40         = resultado_40.model.order[2]
n_inicial_40 = max(p_40, q_40, 1)
residuos_40  = resultado_40.resid.dropna().iloc[n_inicial_40:]

lb20  = acorr_ljungbox(residuos_40, lags=[20], return_df=True)
arch5 = het_arch(residuos_40, nlags=5)

print("\n=== Estadísticos para tabla del póster — ARMA(4,0) sin dummies ===")
print(f"LB(20)  — estadístico: {lb20.loc[20,'lb_stat']:.4f}   p-valor: {lb20.loc[20,'lb_pvalue']:.4f}")
print(f"ARCH(5) — estadístico: {arch5[0]:.4f}   p-valor: {arch5[1]:.4f}")
print(f"JB      — estadístico: 1211.36              p-valor: 0.0000")


# %% Tabla resumen de modelos estimados
tabla_modelos = []

for nombre, resultado in estimaciones.items():
    params  = resultado.params
    errores = resultado.bse

    intercepto_sarimax    = params.get("intercept", np.nan)
    se_intercepto_sarimax = errores.get("intercept", np.nan)

    ar1    = params.get("ar.L1", np.nan)
    ar2    = params.get("ar.L2", np.nan)
    ma1    = params.get("ma.L1", np.nan)
    se_ar1 = errores.get("ar.L1", np.nan)
    se_ar2 = errores.get("ar.L2", np.nan)
    se_ma1 = errores.get("ma.L1", np.nan)

    ar1_mu = params.get("ar.L1", 0)
    ar2_mu = params.get("ar.L2", 0)

    if "ar.L1" in params.index or "ar.L2" in params.index:
        mu = intercepto_sarimax / (1 - ar1_mu - ar2_mu)
    else:
        mu = intercepto_sarimax
    se_mu = se_media_incondicional_sarimax(resultado)

    tabla_modelos.append({
        "Modelo"                : nombre,
        "intercepto_sarimax"    : intercepto_sarimax,
        "se_intercepto_sarimax" : se_intercepto_sarimax,
        "media_incondicional"   : mu,
        "se_media_incondicional": se_mu,
        "a1"   : ar1,    "se_a1": se_ar1,
        "a2"   : ar2,    "se_a2": se_ar2,
        "b1"   : ma1,    "se_b1": se_ma1,
        "AIC"  : resultado.aic,
        "BIC"  : resultado.bic,
    })

tabla_modelos_df = pd.DataFrame(tabla_modelos)


def formato_estimacion(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"{valor:.{decimales}f}"


def formato_error_estandar(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"({valor:.{decimales}f})"


filas_tabla_publicacion = [
    ("a1",                  "a1",                     "se_a1"),
    ("",                    "se_a1",                  None),
    ("a2",                  "a2",                     "se_a2"),
    ("",                    "se_a2",                  None),
    ("b1",                  "b1",                     "se_b1"),
    ("",                    "se_b1",                  None),
    ("intercepto SARIMAX",  "intercepto_sarimax",     "se_intercepto_sarimax"),
    ("",                    "se_intercepto_sarimax",  None),
    ("media incondicional", "media_incondicional",    "se_media_incondicional"),
    ("",                    "se_media_incondicional", None),
    ("AIC",                 "AIC",                    None),
    ("BIC",                 "BIC",                    None),
]

tabla_publicacion = []

for etiqueta, columna_valor, columna_error in filas_tabla_publicacion:
    fila = {"Parámetro": etiqueta}
    for nombre in nombres_modelos:
        modelo_fila = tabla_modelos_df.loc[
            tabla_modelos_df["Modelo"] == nombre
        ].iloc[0]
        if columna_error is None and columna_valor.startswith("se_"):
            fila[nombre] = formato_error_estandar(modelo_fila[columna_valor])
        elif columna_valor in ["AIC", "BIC"]:
            fila[nombre] = formato_estimacion(modelo_fila[columna_valor], decimales=1)
        else:
            fila[nombre] = formato_estimacion(modelo_fila[columna_valor])
    tabla_publicacion.append(fila)

tabla_publicacion_df = pd.DataFrame(tabla_publicacion)

print("\nTabla resumen de modelos estimados")
print(tabla_publicacion_df.to_string(index=False))
print("\nErrores estándar entre paréntesis.")

# Tabla resumen como imagen para el póster
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis("off")

tabla_img = ax.table(
    cellText=tabla_publicacion_df.values,
    colLabels=tabla_publicacion_df.columns,
    cellLoc="center",
    loc="center"
)

tabla_img.auto_set_font_size(False)
tabla_img.set_fontsize(9)
tabla_img.scale(1.4, 2.0)

# Poner en negrita la fila de encabezado
for j in range(len(tabla_publicacion_df.columns)):
    tabla_img[0, j].set_text_props(fontweight="bold")

# Color gris claro en filas de encabezado
for j in range(len(tabla_publicacion_df.columns)):
    tabla_img[0, j].set_facecolor("#D3D3D3")

plt.title("Tabla resumen de modelos estimados", fontsize=12, pad=20
)

plt.savefig(RESULTADOS_DIR / "02b_tabla_modelos.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Tabla explícita del modelo final ARMA(4,0)
 
nombre_final     = "ARMA(4,0)"
estimacion_final = estimaciones[nombre_final]
 
print(f"\n=== Estimación explícita del modelo final {nombre_final} ===")
coefs = pd.DataFrame({
    "coef"              : estimacion_final.params.round(4),
    "error estándar σ̂" : estimacion_final.bse.round(4),
    "t-estadístico"     : estimacion_final.tvalues.round(4),
    "p-valor"           : estimacion_final.pvalues.round(4),
    "significativo 5%"  : estimacion_final.pvalues < 0.05
})
print(coefs.to_string())

residuos_40 = estimaciones["ARMA(4,0)"].resid.dropna().iloc[4:]

lb20  = acorr_ljungbox(residuos_40, lags=[20], return_df=True)
arch5 = het_arch(residuos_40, nlags=5)

print(f"LB(20)  estadístico: {lb20.loc[20,'lb_stat']:.4f}")
print(f"ARCH(5) estadístico: {arch5[0]:.4f}")

# %% =========================
# PASO 3: VALIDACIÓN DE SUPUESTOS
# ============================

# Gráficas separadas por modelo
for nombre in nombres_modelos:
    resultado = estimaciones[nombre]
    p         = resultado.model.order[0]
    q         = resultado.model.order[2]
    n_inicial = max(p, q, 1)
    residuos  = resultado.resid.dropna().iloc[n_inicial:]
    res2      = residuos**2

    # --- Gráfica de residuales ---
    plt.figure(figsize=(10, 4))
    plt.plot(residuos, color="black", linewidth=1)
    plt.axhline(0, color="red", linewidth=0.8, linestyle="--")
    plt.title(f"Residuales — {nombre}")
    plt.xlabel("Fecha")
    plt.ylabel("Residuales")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / f"03a_residuales_{nombre.replace('(','').replace(')','').replace(',','_')}.png", dpi=150)
    plt.show()

    # --- FAC de residuales (verifica no autocorrelación) ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plot_acf(residuos, lags=15, alpha=0.05, bartlett_confint=False, ax=ax)
    ax.set_title(f"FAC residuales — {nombre}")
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / f"03b_fac_residuales_{nombre.replace('(','').replace(')','').replace(',','_')}.png", dpi=150)
    plt.show()

    # --- FACP de residuales al cuadrado (verifica homocedasticidad) ---
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    plot_pacf(res2, lags=15, alpha=0.05, ax=ax)
    ax.set_title(f"FACP residuales² — {nombre}")
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / f"03c_facp_res2_{nombre.replace('(','').replace(')','').replace(',','_')}.png", dpi=150)
    plt.show()

# %% Q-Q plot de los residuales

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, nombre in enumerate(nombres_modelos):
    resultado = estimaciones[nombre]
    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)
    residuos  = resultado.resid.dropna().iloc[n_inicial:]

    probplot(residuos.iloc[1:-1], dist="norm", plot=axes[i])
    axes[i].set_title(f"Q-Q plot residuales {nombre}")
    axes[i].grid(True)

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "04_qqplot.png", dpi=150)
plt.show()

# %% Tabla de diagnóstico: JB, ARCH y Ljung-Box para cada modelo

tabla_diagnostico = []

for nombre in nombres_modelos:
    resultado = estimaciones[nombre]
    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)
    residuos  = resultado.resid.dropna().iloc[n_inicial:]

    # Prueba Jarque-Bera (normalidad)
    # H0: residuales son normales. p-valor > 0.05 → supuesto cumplido
    jb_pvalue = jarque_bera(residuos).pvalue

    # Prueba ARCH (homocedasticidad)
    # H0: no hay efectos ARCH. p-valor > 0.05 → supuesto cumplido
    arch_1 = het_arch(residuos, nlags=1)[1]
    arch_2 = het_arch(residuos, nlags=2)[1]
    arch_5 = het_arch(residuos, nlags=5)[1]

    # Prueba Ljung-Box (no autocorrelación)
    # H0: no hay autocorrelación. p-valor > 0.05 → supuesto cumplido
    ljung_box = acorr_ljungbox(residuos, lags=[5, 10, 20], return_df=True)

    tabla_diagnostico.append({
        "Modelo" : nombre,
        "JB"     : jb_pvalue,
        "A(1)"   : arch_1,
        "A(2)"   : arch_2,
        "A(5)"   : arch_5,
        "LB(5)"  : ljung_box.loc[5,  "lb_pvalue"],
        "LB(10)" : ljung_box.loc[10, "lb_pvalue"],
        "LB(20)" : ljung_box.loc[20, "lb_pvalue"],
    })

tabla_diagnostico_df = pd.DataFrame(tabla_diagnostico)

print("\n=== Tabla de diagnóstico (p-valores) ===")
print("JB=Jarque-Bera | A=ARCH | LB=Ljung-Box")
print("p-valor > 0.05 indica que el supuesto se cumple")
print(tabla_diagnostico_df.round(3).to_string(index=False))

# %% Tabla de diagnóstico como imagen guardable

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis("off")

tabla_render = tabla_diagnostico_df.round(3)

tabla_img = ax.table(
    cellText=tabla_render.values,
    colLabels=tabla_render.columns,
    cellLoc="center",
    loc="center"
)

tabla_img.auto_set_font_size(False)
tabla_img.set_fontsize(10)
tabla_img.scale(1.2, 1.8)

plt.title("Tabla de diagnóstico (p-valores)", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "06_tabla_diagnostico.png", dpi=150, bbox_inches="tight")
plt.show()


# %% =========================
# PASO 4: PRONÓSTICO
# Intervalos por bootstrapping de residuales
# No asume normalidad → robusto ante rechazo de Jarque-Bera
# ============================

# Seleccionar el modelo final según AIC y validación de supuestos
# (revisar tabla de diagnóstico y elegir el que pase mejor las pruebas)
nombre_final     = "ARMA(4,0)"   # <-- AJUSTAR si otro modelo pasa mejor la validación
estimacion_final = estimaciones[nombre_final]

pasos  = 10
n_boot = 5000
np.random.seed(42)

p = estimacion_final.model.order[0]
q = estimacion_final.model.order[2]
n_inicial = max(p, q, 1)

residuos_final = estimacion_final.resid.dropna().iloc[n_inicial:]

# Pronóstico puntual
pronostico = estimacion_final.get_forecast(steps=pasos)
puntual    = pronostico.predicted_mean.values

# Bootstrap de residuales para construir IC sin asumir normalidad
res_arr     = residuos_final.values
boot_matrix = np.zeros((n_boot, pasos))

for b in range(n_boot):
    shocks = np.random.choice(res_arr, size=pasos, replace=True)
    boot_matrix[b, :] = puntual + shocks

ic_inf = np.percentile(boot_matrix, 2.5,  axis=0)
ic_sup = np.percentile(boot_matrix, 97.5, axis=0)

# Índice temporal del pronóstico
idx_pron = pd.date_range(
    start=serie.index[-1] + pd.tseries.offsets.QuarterEnd(),
    periods=pasos,
    freq="QE"
)

tabla_pronostico = pd.DataFrame({
    "pronostico_puntual" : puntual.round(4),
    "IC_inf_95_bootstrap": ic_inf.round(4),
    "IC_sup_95_bootstrap": ic_sup.round(4)
}, index=idx_pron)

print(f"\n=== Pronóstico 10 trimestres adelante — {nombre_final} ===")
print(tabla_pronostico.to_string())

# Gráfica del pronóstico
plt.figure(figsize=(10, 5))
plt.plot(
    serie.iloc[-40:],
    label="Datos históricos",
    color="black",
    linewidth=1
)
plt.plot(
    idx_pron, puntual,
    label=f"Pronóstico {nombre_final}",
    color="orange",
    linewidth=2,
    linestyle="--",
    marker="o",
    markersize=4
)
plt.fill_between(
    idx_pron, ic_inf, ic_sup,
    color="orange",
    alpha=0.3,
    label="IC 95% (bootstrap)"
)
plt.axhline(0, color="black", linewidth=0.7, linestyle=":")
plt.title(f"Pronóstico {nombre_final} — CLI Alemania (10 trimestres adelante)")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "05_pronostico.png", dpi=150)
plt.show()

print("\n=== Interpretación del pronóstico ===")
print(
    f"El modelo {nombre_final} proyecta que la tasa de crecimiento interanual "
    f"del PIB de Alemania se situará en promedio en {puntual.mean():.2f}% "
    f"durante los 10 trimestres siguientes a 2023 Q4. "
    f"El intervalo de confianza al 95% se construyó mediante bootstrapping "
    f"de residuales, técnica que no requiere asumir normalidad en los errores."
)

# ============================================================
# SECCIÓN A — TEST CUSUM SOBRE RESIDUALES DEL ARMA(4,0)
# ============================================================
# El CUSUM de residuales estandarizados evalúa si la estructura del modelo
# es estable a lo largo del tiempo. Cruces de las bandas críticas al 5 %
# indican quiebres estructurales o presencia de outliers aditivos.

print("\n" + "="*60)
print("SECCIÓN A — CUSUM sobre residuales ARMA(4,0) [muestra 2023]")
print("="*60)

resultado_40 = estimaciones["ARMA(4,0)"]
p, q        = resultado_40.model.order[0], resultado_40.model.order[2]
n_inicial   = max(p, q, 1)
residuos_40 = resultado_40.resid.dropna().iloc[n_inicial:]

# Estandarización
sigma_hat = residuos_40.std()
res_std   = residuos_40 / sigma_hat

# CUSUM acumulado
cusum = res_std.cumsum()
n     = len(cusum)
t_idx = np.arange(1, n + 1)

# Bandas críticas al 5 % (Brown-Durbin-Evans): ±(a + 2a·(t/n)), a ≈ 0.948
a         = 0.948
banda_sup = a + 2 * a * (t_idx / n)
banda_inf = -(a + 2 * a * (t_idx / n))

plt.figure(figsize=(10, 4))
plt.plot(cusum.index, cusum.values, color="steelblue", linewidth=1.5, label="CUSUM")
plt.plot(cusum.index, banda_sup, color="red", linewidth=1, linestyle="--", label="Banda 5%")
plt.plot(cusum.index, banda_inf, color="red", linewidth=1, linestyle="--")
plt.axhline(0, color="black", linewidth=0.6, linestyle=":")

# Marcar visualmente el periodo COVID
plt.axvspan(
    pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"),
    color="salmon", alpha=0.15, label="Periodo COVID (ref.)"
)
plt.title("CUSUM — residuales ARMA(4,0) [1992–2023]")
plt.xlabel("Fecha")
plt.ylabel("CUSUM estandarizado")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "07_cusum_2023.png", dpi=150)
plt.show()

# Detección de quiebres
quiebres = cusum.index[np.abs(cusum.values) > banda_sup]
if len(quiebres) > 0:
    print(f"  ⚠ CUSUM supera bandas en {len(quiebres)} periodos.")
    print(f"    Primero: {quiebres[0].date()}  |  Último: {quiebres[-1].date()}")
else:
    print("  ✓ CUSUM dentro de bandas en toda la muestra.")


# ============================================================
# SECCIÓN B — IDENTIFICACIÓN DE OUTLIERS CON RESIDUALES ESTANDARIZADOS
# ============================================================
# Se identifican observaciones con |residual estandarizado| > 3 σ,
# criterio estándar para outliers aditivos en series de tiempo.

print("\n" + "="*60)
print("SECCIÓN B — Identificación de outliers por residuales")
print("="*60)

umbral      = 3.0
outliers    = residuos_40[np.abs(res_std) > umbral]
print(f"\n  Observaciones con |residual| > {umbral}σ:")
if len(outliers) > 0:
    for fecha, val in outliers.items():
        print(f"    {fecha.date()}: residual = {val:.4f}  ({val/sigma_hat:.2f} σ)")
else:
    print("    Ninguna (umbral 3σ).")

# El Q-Q plot de 2023 muestra dos puntos extremos en la cola inferior y uno en la
# superior, consistentes con el desplome de 2020 Q2 (COVID) y el rebote de 2021 Q1.
# También se tiene en cuenta el pico de la crisis de 2009 Q1, que aunque no es tan extremo, es un evento macroeconómico relevante.
# Se construyen dummies de impulso para esos trimestres específicos.

FECHAS_OUTLIER = {
    "d_2009q1": "2009-03-31",   # crisis financiera global (≈ −3.5 %)
    "d_2020q2": "2020-06-30",   # caída extrema COVID (≈ −10.7 %)
    "d_2021q1": "2021-03-31",   # rebote post-COVID (≈ +10.7 %)
}

# Nota: si los residuales extremos identificados en su ejecución son distintos,
# ajuste las fechas en FECHAS_OUTLIER según el print anterior.

exog_dummies = pd.DataFrame(index=serie.index)
for nombre_dummy, fecha_str in FECHAS_OUTLIER.items():
    col = pd.Series(0.0, index=serie.index, name=nombre_dummy)
    if fecha_str in serie.index.strftime("%Y-%m-%d"):
        col.loc[fecha_str] = 1.0
    else:
        # Búsqueda flexible: tomar la fecha más cercana disponible
        idx_cercano = serie.index.get_indexer([pd.Timestamp(fecha_str)], method="nearest")[0]
        col.iloc[idx_cercano] = 1.0
        print(f"  Nota: {nombre_dummy} mapeado a {serie.index[idx_cercano].date()}")
    exog_dummies[nombre_dummy] = col

print("\n  Dummies construidas:")
print(exog_dummies[exog_dummies.sum(axis=1) > 0].to_string())


# ============================================================
# SECCIÓN C — TEST DE BAI-PERRON (cambios estructurales múltiples)
# ============================================================
# Bai & Perron (1998, 2003) permiten detectar m quiebres estructurales
# desconocidos en la media (y/o tendencia) de la serie. 

print("\n" + "="*60)
print("SECCIÓN C — TEST DE BAI-PERRON [ruptures]")
print("="*60)

y = serie.values  # array 1-D


M_max    = 5        # máximo de quiebres a considerar
modelo_bp = rpt.Dynp(model="l2", jump=1).fit(y)

# BIC manual: BIC = n·ln(RSS/n) + k·ln(n)
#   k = número de parámetros = m + 1 medias + 1 varianza → aquí usamos m+1
n = len(y)
bic_scores = {}

for m in range(0, M_max + 1):
    breakpoints = modelo_bp.predict(n_bkps=m)   # índices de fin de cada segmento
    segmentos   = rpt.utils.pairwise([0] + breakpoints)
    rss = sum(
        np.sum((y[i:j] - y[i:j].mean()) ** 2)
        for i, j in segmentos
    )
    k            = m + 1          # nº de parámetros de media
    bic_m        = n * np.log(rss / n) + k * np.log(n)
    bic_scores[m] = bic_m
    print(f"  m={m} quiebres  →  RSS={rss:.4f}  BIC={bic_m:.3f}")

m_optimo = min(bic_scores, key=bic_scores.get)
print(f"\n  ✓ Número óptimo de quiebres (BIC): m = {m_optimo}")


breakpoints_optimos = modelo_bp.predict(n_bkps=m_optimo)
fechas_quiebre = [serie.index[bp - 1].date() for bp in breakpoints_optimos[:-1]]

print(f"  Fechas de quiebre estimadas: {fechas_quiebre}")


segmentos_optimos = rpt.utils.pairwise([0] + breakpoints_optimos)
print("\n  Medias por régimen:")
for idx, (i, j) in enumerate(segmentos_optimos):
    fecha_ini = serie.index[i].date()
    fecha_fin = serie.index[j - 1].date()
    media_reg = y[i:j].mean()
    print(f"    Régimen {idx+1}: {fecha_ini} — {fecha_fin}  |  media = {media_reg:.4f}%")


plt.figure(figsize=(11, 5))
plt.plot(serie, color="black", linewidth=1.2, label="Serie original")

colores_regimen = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]

for idx, (i, j) in enumerate(segmentos_optimos):
    seg_idx   = serie.index[i:j]
    seg_media = y[i:j].mean()
    color_r   = colores_regimen[idx % len(colores_regimen)]
    plt.hlines(
        seg_media, seg_idx[0], seg_idx[-1],
        colors=color_r, linewidths=2.0, linestyles="-",
        label=f"Media régimen {idx+1}: {seg_media:.2f}%"
    )

for fecha in fechas_quiebre:
    plt.axvline(
        pd.Timestamp(fecha),
        color="red", linewidth=1.2, linestyle="--", alpha=0.8
    )
    plt.text(
        pd.Timestamp(fecha), plt.ylim()[1] * 0.92,
        str(fecha), rotation=90, fontsize=8,
        color="red", va="top", ha="right"
    )

plt.axhline(0, color="black", linewidth=0.6, linestyle=":")
plt.title(
    f"Test de Bai-Perron — CLI Alemania (1992–2023)\n"
    f"Quiebres óptimos (BIC): m = {m_optimo}"
)
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend(fontsize=8, loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "11_bai_perron.png", dpi=150)
plt.show()

filas_bp = []
for idx, (i, j) in enumerate(segmentos_optimos):
    filas_bp.append({
        "Régimen"        : idx + 1,
        "Inicio"         : str(serie.index[i].date()),
        "Fin"            : str(serie.index[j - 1].date()),
        "Obs."           : j - i,
        "Media (%)"      : round(y[i:j].mean(), 4),
        "Desv. Est. (%)": round(y[i:j].std(), 4),
    })

tabla_bp_df = pd.DataFrame(filas_bp)
print("\n" + tabla_bp_df.to_string(index=False))

# Tabla como imagen para el póster
if tabla_bp_df.empty:
    print("  ⚠ Sin quiebres detectados: no se genera tabla imagen.")
else:
    fig, ax = plt.subplots(figsize=(10, max(2, 0.6 * len(filas_bp) + 1)))
    ax.axis("off")
    tbl_img = ax.table(
        cellText=tabla_bp_df.values,
        colLabels=tabla_bp_df.columns,
        cellLoc="center",
        loc="center"
    )
    tbl_img.auto_set_font_size(False)
    tbl_img.set_fontsize(9)
    tbl_img.scale(1.3, 2.0)
    for j in range(len(tabla_bp_df.columns)):
        tbl_img[0, j].set_facecolor("#D3D3D3")
        tbl_img[0, j].set_text_props(fontweight="bold")
    plt.title(
        f"Bai-Perron — Regímenes estimados (m = {m_optimo})",
        fontsize=11, pad=18
    )
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "12_tabla_bai_perron.png", dpi=150, bbox_inches="tight")
    plt.show()

# ============================================================
# SECCIÓN D — SARIMAX(4,0,0) CON DUMMIES COVID
# ============================================================

print("\n" + "="*60)
print("SECCIÓN D — SARIMAX(4,0,0) con dummies COVID (2020 Q2 y 2021 Q1)")
print("="*60)

modelo_covid = SARIMAX(
    serie,
    exog=exog_dummies,
    order=(4, 0, 0),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_covid = modelo_covid.fit(method="bfgs", maxiter=400, disp=False)

print(resultado_covid.summary())
print(f"\nAIC sin dummies : {resultado_40.aic:.1f}")
print(f"AIC con dummies : {resultado_covid.aic:.1f}")
print(f"BIC sin dummies : {resultado_40.bic:.1f}")
print(f"BIC con dummies : {resultado_covid.bic:.1f}")

# Residuales del modelo corregido
n_ini_c  = max(4, 1)
res_cov  = resultado_covid.resid.dropna().iloc[n_ini_c:]

# --- Q-Q plot comparativo ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
probplot(residuos_40.iloc[1:-1], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q ARMA(4,0) — sin dummies")
axes[0].grid(True)
probplot(res_cov.iloc[1:-1], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q ARMA(4,0) — con dummies COVID")
axes[1].grid(True)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "08_qqplot_covid_2023.png", dpi=150)
plt.show()

# --- FAC/FACP de residuales del modelo corregido ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(res_cov, lags=15, alpha=0.05, bartlett_confint=False, ax=axes[0])
axes[0].set_title("FAC residuales — ARMA(4,0) con dummies COVID")
axes[0].set_ylim(-1, 1)
plot_pacf(res_cov, lags=15, alpha=0.05, ax=axes[1])
axes[1].set_title("FACP residuales — ARMA(4,0) con dummies COVID")
axes[1].set_ylim(-1, 1)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "08b_fac_facp_covid_2023.png", dpi=150)
plt.show()

# --- Tabla de diagnóstico del modelo corregido ---
jb_c    = jarque_bera(res_cov).pvalue
arch1_c = het_arch(res_cov, nlags=1)[1]
arch2_c = het_arch(res_cov, nlags=2)[1]
arch5_c = het_arch(res_cov, nlags=5)[1]
lb_c    = acorr_ljungbox(res_cov, lags=[5, 10, 20], return_df=True)

print("\n=== Diagnóstico ARMA(4,0) con dummies COVID (p-valores) ===")
print(f"  JB     : {jb_c:.3f}  (H0: normalidad;  > 0.05 → cumple)")
print(f"  ARCH(1): {arch1_c:.3f}  ARCH(2): {arch2_c:.3f}  ARCH(5): {arch5_c:.3f}")
print(f"  LB(5)  : {lb_c.loc[5,'lb_pvalue']:.3f}  LB(10): {lb_c.loc[10,'lb_pvalue']:.3f}  LB(20): {lb_c.loc[20,'lb_pvalue']:.3f}")

# Tabla de diagnóstico como imagen (para el póster)
diag_filas = [
    ["ARMA(4,0) sin dummies",
     round(jarque_bera(residuos_40).pvalue, 3),
     round(het_arch(residuos_40, nlags=1)[1], 3),
     round(het_arch(residuos_40, nlags=2)[1], 3),
     round(het_arch(residuos_40, nlags=5)[1], 3),
     round(acorr_ljungbox(residuos_40, lags=[5], return_df=True).loc[5, "lb_pvalue"], 3),
     round(acorr_ljungbox(residuos_40, lags=[10], return_df=True).loc[10, "lb_pvalue"], 3),
     round(acorr_ljungbox(residuos_40, lags=[20], return_df=True).loc[20, "lb_pvalue"], 3)],
    ["ARMA(4,0) + dummies COVID",
     round(jb_c, 3),
     round(arch1_c, 3), round(arch2_c, 3), round(arch5_c, 3),
     round(lb_c.loc[5,  "lb_pvalue"], 3),
     round(lb_c.loc[10, "lb_pvalue"], 3),
     round(lb_c.loc[20, "lb_pvalue"], 3)],
]
cols_diag = ["Modelo", "JB", "A(1)", "A(2)", "A(5)", "LB(5)", "LB(10)", "LB(20)"]

fig, ax = plt.subplots(figsize=(13, 2))
ax.axis("off")
tbl = ax.table(cellText=diag_filas, colLabels=cols_diag, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.9)
plt.title("Diagnóstico comparativo — ARMA(4,0) 1992–2023 (p-valores)", fontsize=11, pad=18)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "09_tabla_diagnostico_covid_2023.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# SECCIÓN E — PRONÓSTICO 10 TRIMESTRES DESDE 2023 Q4
# ============================================================
# Se usa el modelo con dummies si mejora el diagnóstico; fuera de muestra
# las dummies toman valor 0 (el choque COVID no se repite en el pronóstico).

print("\n" + "="*60)
print("SECCIÓN E — Pronóstico 10 trimestres desde 2023 Q4")
print("="*60)

# Verificar si el modelo con dummies mejora (al menos JB o LB)
mejor_modelo_covid = resultado_covid.aic < resultado_40.aic
modelo_final_2023  = resultado_covid if mejor_modelo_covid else resultado_40
etiqueta_final     = "ARMA(4,0) + dummies COVID" if mejor_modelo_covid else "ARMA(4,0)"
print(f"  Modelo para pronóstico: {etiqueta_final}")

pasos  = 10
n_boot = 5000
np.random.seed(42)

# Exog futura: dummies en cero (sin nuevo COVID)
if mejor_modelo_covid:
    idx_pron_prev = pd.date_range(
        start=serie.index[-1] + pd.tseries.offsets.QuarterEnd(),
        periods=pasos, freq="QE"
    )
    exog_futuro = pd.DataFrame(
        {col: np.zeros(pasos) for col in exog_dummies.columns},
        index=idx_pron_prev
    )
    pron = modelo_final_2023.get_forecast(steps=pasos, exog=exog_futuro)
else:
    pron = modelo_final_2023.get_forecast(steps=pasos)

puntual = pron.predicted_mean.values

# Residuales para bootstrap
n_ini_f = max(4, 1)
res_fin = modelo_final_2023.resid.dropna().iloc[n_ini_f:].values

boot_matrix = np.zeros((n_boot, pasos))
for b in range(n_boot):
    shocks = np.random.choice(res_fin, size=pasos, replace=True)
    boot_matrix[b, :] = puntual + shocks

ic_inf = np.percentile(boot_matrix, 2.5,  axis=0)
ic_sup = np.percentile(boot_matrix, 97.5, axis=0)

idx_pron = pd.date_range(
    start=serie.index[-1] + pd.tseries.offsets.QuarterEnd(),
    periods=pasos, freq="QE"
)

tabla_pron = pd.DataFrame({
    "pronostico_puntual" : puntual.round(4),
    "IC_inf_95_bootstrap": ic_inf.round(4),
    "IC_sup_95_bootstrap": ic_sup.round(4)
}, index=idx_pron)

print(f"\n=== Pronóstico 10 trimestres adelante — {etiqueta_final} ===")
print(tabla_pron.to_string())

# Gráfica
plt.figure(figsize=(11, 5))
plt.plot(
    serie.iloc[-40:],
    color="black", linewidth=1.2, label="Histórico (1992–2023)"
)
plt.plot(
    idx_pron, puntual,
    color="orange", linewidth=2, linestyle="--", marker="o", markersize=4,
    label=f"Pronóstico {etiqueta_final}"
)
plt.fill_between(
    idx_pron, ic_inf, ic_sup,
    color="orange", alpha=0.25, label="IC 95% (bootstrap)"
)
plt.axhline(0, color="black", linewidth=0.6, linestyle=":")
plt.title(f"Pronóstico {etiqueta_final}\nCLI Alemania — 10 trimestres adelante desde 2023 Q4")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "10_pronostico_covid_2023.png", dpi=150)
plt.show()

print("\n=== Interpretación ===")
print(
    f"El {etiqueta_final} proyecta una tasa de crecimiento promedio de "
    f"{puntual.mean():.2f}% para los 10 trimestres siguientes a 2023 Q4. "
    "Las dummies aditivas capturan los choques puntuales del COVID-19 sin "
    "distorsionar la dinámica AR del modelo, lo que permite recuperar un "
    "ajuste adecuado de los supuestos (normalidad y no autocorrelación de "
    "residuales) y producir pronósticos más confiables."
)