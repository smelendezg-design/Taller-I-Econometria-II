# Taller 1 Poster — Análisis de la tasa de crecimiento del PIB de Alemania (1992–2019)
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
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson

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

# Se excluye el periodo COVID (2020 en adelante) por ser un choque externo
# atípico que viola los supuestos del modelo ARMA: los residuales presentan
# heterocedasticidad, autocorrelación y no normalidad en todos los modelos
# candidatos cuando se incluye dicho periodo.
# Se excluye de la serie de tiempo los periodos anteriores al 1992-06-30, ya que concuerda
# con los choques sobre el PIB derivado de la reunificación alemana. Este periodo fue confirmado
# con la metodología para detectar cambios estructurales en la serie de tiempo Test de Chow. 
# Se trabaja con 1961 Q1 - 2019 Q4 (111 observaciones).

serie = serie["1992-06-30":"2019-12-31"]

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
plt.title("CLI: tasa de crecimiento del PIB de Alemania (1992–2019)")
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

    # --- FAC y FACP de los residuales (verifica no autocorrelación) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(residuos, lags=15, alpha=0.05, bartlett_confint=False, ax=axes[0])
    axes[0].set_title(f"FAC residuales — {nombre}")
    axes[0].set_ylim(-1, 1)
    plot_pacf(residuos, lags=15, alpha=0.05, ax=axes[1])
    axes[1].set_title(f"FACP residuales — {nombre}")
    axes[1].set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / f"03b_fac_facp_residuales_{nombre.replace('(','').replace(')','').replace(',','_')}.png", dpi=150)
    plt.show()

    # --- FAC y FACP de los residuales al cuadrado (verifica homocedasticidad) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(res2, lags=15, alpha=0.05, bartlett_confint=False, ax=axes[0])
    axes[0].set_title(f"FAC residuales² — {nombre}")
    axes[0].set_ylim(-1, 1)
    plot_pacf(res2, lags=15, alpha=0.05, ax=axes[1])
    axes[1].set_title(f"FACP residuales² — {nombre}")
    axes[1].set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / f"03c_fac_facp_res2_{nombre.replace('(','').replace(')','').replace(',','_')}.png", dpi=150)
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
    f"durante los 10 trimestres siguientes a 2019 Q4. "
    f"El intervalo de confianza al 95% se construyó mediante bootstrapping "
    f"de residuales, técnica que no requiere asumir normalidad en los errores."
)

# ============================================================
#  PRONÓSTICO
#  SECCIÓN A — TEST CUSUM SOBRE RESIDUALES DEL ARMA(4,0)
# ============================================================
# El CUSUM (suma acumulada de residuales estandarizados) evalúa si los
# parámetros del modelo son estables a lo largo del tiempo.
# Si la suma acumulada supera las bandas críticas al 5 % → inestabilidad.
#
# Referencia: Brown, Durbin & Evans (1975), JRSS-B.

print("\n" + "="*60)
print("SECCIÓN A — CUSUM sobre residuales ARMA(4,0) [muestra 2019]")
print("="*60)

resultado_40 = estimaciones["ARMA(4,0)"]
p, q        = resultado_40.model.order[0], resultado_40.model.order[2]
n_inicial   = max(p, q, 1)
residuos_40 = resultado_40.resid.dropna().iloc[n_inicial:]

# Estandarización: dividir por la desviación estándar muestral de los residuales
sigma_hat = residuos_40.std()
res_std   = residuos_40 / sigma_hat

# CUSUM acumulado
cusum = res_std.cumsum()
n     = len(cusum)
t_idx = np.arange(1, n + 1)

# Bandas críticas al 5 % (Brown-Durbin-Evans):  ±(a + 2a·(t/n))  con a ≈ 0.948
a       = 0.948
banda_sup = a + 2 * a * (t_idx / n)
banda_inf = -(a + 2 * a * (t_idx / n))

plt.figure(figsize=(10, 4))
plt.plot(cusum.index, cusum.values, color="steelblue", linewidth=1.5, label="CUSUM")
plt.plot(cusum.index, banda_sup, color="red", linewidth=1, linestyle="--", label="Banda 5%")
plt.plot(cusum.index, banda_inf, color="red", linewidth=1, linestyle="--")
plt.axhline(0, color="black", linewidth=0.6, linestyle=":")
plt.title("CUSUM — residuales ARMA(4,0) [1992–2019]")
plt.xlabel("Fecha")
plt.ylabel("CUSUM estandarizado")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "07_cusum_2019.png", dpi=150)
plt.show()

# Detección automática de quiebres: periodos donde |CUSUM| > banda
quiebres = cusum.index[np.abs(cusum.values) > banda_sup]
if len(quiebres) > 0:
    print(f"  ⚠ CUSUM supera bandas en {len(quiebres)} periodos.")
    print(f"    Primero: {quiebres[0].date()}  |  Último: {quiebres[-1].date()}")
    cusum_inestable = True
else:
    print("  ✓ CUSUM dentro de bandas → parámetros estables en toda la muestra.")
    cusum_inestable = False


# ============================================================
#  PRONOSTICO
# SECCIÓN B — SARIMAX CON DUMMY DE OUTLIER (si CUSUM lo sugiere)
# ============================================================
# El Q-Q plot de 2019 muestra un residual extremo en el extremo inferior
# que coincide con la Gran Recesión (2008 Q4: −6.7 %).
# Si el CUSUM también señala inestabilidad alrededor de ese periodo,
# se incluye una variable dummy aditiva para ese trimestre.
# Una dummy aditiva (impulso) modela un choque puntual sin afectar la
# dinámica AR del modelo; es la corrección estándar para outliers aditivos
# en series de tiempo (Chang, Chen & Tiao, 1988).

print("\n" + "="*60)
print("SECCIÓN B — ARMA(4,0) con dummy outlier 2008 Q4")
print("="*60)

# Fecha del outlier identificado en el Q-Q plot (cola inferior extrema)
FECHA_OUTLIER_2008 = "2008-12-31"

dummy_2008           = pd.Series(0.0, index=serie.index, name="d_2008q4")
dummy_2008.loc[FECHA_OUTLIER_2008] = 1.0

# Si el CUSUM detectó inestabilidad adicional en 2019 (último dato), se puede
# añadir otra dummy. Por defecto sólo corregimos 2008 Q4.
exog_dummy = dummy_2008.to_frame()

modelo_dummy = SARIMAX(
    serie,
    exog=exog_dummy,
    order=(4, 0, 0),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
)
resultado_dummy = modelo_dummy.fit(method="bfgs", maxiter=400, disp=False)

print(resultado_dummy.summary())
print(f"\nAIC con dummy : {resultado_dummy.aic:.1f}")
print(f"AIC sin dummy : {resultado_40.aic:.1f}")
print(f"BIC con dummy : {resultado_dummy.bic:.1f}")
print(f"BIC sin dummy : {resultado_40.bic:.1f}")

# Residuales del modelo con dummy
p_d, q_d    = 4, 0
n_ini_d     = max(p_d, q_d, 1)
res_dummy   = resultado_dummy.resid.dropna().iloc[n_ini_d:]

# Q-Q plot comparativo: sin dummy vs con dummy
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
probplot(residuos_40.iloc[1:-1], dist="norm", plot=axes[0])
axes[0].set_title("Q-Q residuales ARMA(4,0) — sin dummy")
axes[0].grid(True)
probplot(res_dummy.iloc[1:-1], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q residuales ARMA(4,0) — con dummy 2008 Q4")
axes[1].grid(True)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "08_qqplot_dummy_2019.png", dpi=150)
plt.show()

# Tabla de diagnóstico del modelo con dummy
jb_d    = jarque_bera(res_dummy).pvalue
arch1_d = het_arch(res_dummy, nlags=1)[1]
arch2_d = het_arch(res_dummy, nlags=2)[1]
arch5_d = het_arch(res_dummy, nlags=5)[1]
lb_d    = acorr_ljungbox(res_dummy, lags=[5, 10, 20], return_df=True)

print("\n=== Diagnóstico ARMA(4,0) con dummy 2008 Q4 (p-valores) ===")
print(f"  JB     : {jb_d:.3f}  (H0: normalidad)")
print(f"  ARCH(1): {arch1_d:.3f}  ARCH(2): {arch2_d:.3f}  ARCH(5): {arch5_d:.3f}")
print(f"  LB(5)  : {lb_d.loc[5,'lb_pvalue']:.3f}  LB(10): {lb_d.loc[10,'lb_pvalue']:.3f}  LB(20): {lb_d.loc[20,'lb_pvalue']:.3f}")
print("  p-valor > 0.05 → supuesto cumplido")


# ============================================================
# PRONÓSTICO
# SECCIÓN C — PRONÓSTICO 2020–2023 Y COMPARACIÓN CON DATOS REALES
# ============================================================
# Se usa el mejor modelo (con o sin dummy según diagnóstico) para pronosticar
# 16 trimestres desde 2020 Q1, y se compara con la serie observada 2020–2023.

print("\n" + "="*60)
print("SECCIÓN C — Pronóstico 2020–2023 vs datos reales")
print("="*60)

# Decidir qué modelo usar: con dummy si mejora AIC/BIC y diagnóstico
usar_dummy = resultado_dummy.aic < resultado_40.aic
modelo_seleccionado = resultado_dummy if usar_dummy else resultado_40
etiqueta_modelo = "ARMA(4,0)+dummy 2008Q4" if usar_dummy else "ARMA(4,0)"
print(f"  Modelo seleccionado para pronóstico: {etiqueta_modelo}")

# Cargar serie completa hasta 2023 para comparar
BASE_DIR_EXT = Path(__file__).resolve().parent
raw_full     = pd.read_excel(
    BASE_DIR_EXT / "datos" / "LORSGPORDEQ659S.xlsx",
    sheet_name="Quarterly",
    parse_dates=["observation_date"]
)
raw_full = raw_full.set_index("observation_date")
raw_full.index = pd.DatetimeIndex(raw_full.index).to_period("Q").to_timestamp("Q")
serie_full  = pd.to_numeric(raw_full["LORSGPORDEQ659S"], errors="coerce").dropna()
serie_real  = serie_full["2020-01-01":"2023-12-31"]  # 16 observaciones reales

pasos_comp  = len(serie_real)   # tantos pasos como observaciones reales disponibles
n_boot_comp = 5000
np.random.seed(42)

# Pronóstico puntual desde el último dato de 2019
if usar_dummy:
    # Exog futuro: fuera de muestra, la dummy toma valor 0
    exog_futuro = pd.DataFrame(
        {"d_2008q4": np.zeros(pasos_comp)},
        index=pd.date_range(
            start=serie.index[-1] + pd.tseries.offsets.QuarterEnd(),
            periods=pasos_comp, freq="QE"
        )
    )
    pron_comp = modelo_seleccionado.get_forecast(steps=pasos_comp, exog=exog_futuro)
else:
    pron_comp = modelo_seleccionado.get_forecast(steps=pasos_comp)

puntual_comp = pron_comp.predicted_mean.values

# Bootstrap IC
p_sel = modelo_seleccionado.model.order[0]
q_sel = modelo_seleccionado.model.order[2]
n_ini_sel = max(p_sel, q_sel, 1)
res_sel   = modelo_seleccionado.resid.dropna().iloc[n_ini_sel:].values

boot_comp = np.zeros((n_boot_comp, pasos_comp))
for b in range(n_boot_comp):
    shocks = np.random.choice(res_sel, size=pasos_comp, replace=True)
    boot_comp[b, :] = puntual_comp + shocks

ic_inf_comp = np.percentile(boot_comp, 2.5,  axis=0)
ic_sup_comp = np.percentile(boot_comp, 97.5, axis=0)

idx_pron_comp = pd.date_range(
    start=serie.index[-1] + pd.tseries.offsets.QuarterEnd(),
    periods=pasos_comp, freq="QE"
)

# --- Tabla comparativa pronóstico vs real ---
tabla_comp = pd.DataFrame({
    "pronostico"  : puntual_comp.round(4),
    "real"        : serie_real.values,
    "IC_inf_95"   : ic_inf_comp.round(4),
    "IC_sup_95"   : ic_sup_comp.round(4),
    "dentro_IC"   : (serie_real.values >= ic_inf_comp) & (serie_real.values <= ic_sup_comp)
}, index=idx_pron_comp)

print("\n=== Pronóstico vs datos reales 2020–2023 ===")
print(tabla_comp.to_string())
pct_dentro = tabla_comp["dentro_IC"].mean() * 100
print(f"\n  {pct_dentro:.0f}% de los valores reales cae dentro del IC 95% (bootstrap).")

# --- Gráfica comparativa ---
n_hist = 40   # últimos 40 trimestres de la muestra de estimación para contexto
plt.figure(figsize=(11, 5))
plt.plot(
    serie.iloc[-n_hist:],
    color="black", linewidth=1.2, label="Histórico (hasta 2019 Q4)"
)
plt.plot(
    idx_pron_comp, puntual_comp,
    color="orange", linewidth=2, linestyle="--", marker="o", markersize=4,
    label=f"Pronóstico {etiqueta_modelo}"
)
plt.fill_between(
    idx_pron_comp, ic_inf_comp, ic_sup_comp,
    color="orange", alpha=0.25, label="IC 95% (bootstrap)"
)
plt.plot(
    serie_real.index, serie_real.values,
    color="royalblue", linewidth=1.5, marker="s", markersize=4,
    label="Dato real 2020–2023"
)
plt.axhline(0, color="black", linewidth=0.6, linestyle=":")
plt.title(f"Pronóstico {etiqueta_modelo} vs datos reales 2020–2023")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "09_pronostico_vs_real_2019.png", dpi=150)
plt.show()

# Métricas de error de pronóstico
errores_pron = serie_real.values - puntual_comp
mae  = np.mean(np.abs(errores_pron))
rmse = np.sqrt(np.mean(errores_pron**2))
print(f"\n  MAE  del pronóstico: {mae:.4f}")
print(f"  RMSE del pronóstico: {rmse:.4f}")
print(
    "\n  Nota: el COVID-19 (2020 Q2) genera un choque de magnitud excepcional "
    "que ningún modelo ARMA puede anticipar. El RMSE elevado refleja ese evento "
    "y no un mal ajuste del modelo al patrón cíclico normal de la serie."
)