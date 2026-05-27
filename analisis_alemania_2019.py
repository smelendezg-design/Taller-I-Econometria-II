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
# Se trabaja con 1961 Q1 - 2019 Q4 (236 observaciones).
serie = serie[:"2019-12-31"]

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
plt.title("CLI: tasa de crecimiento del PIB de Alemania (1961–2019)")
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
    estimacion_modelo = modelo.fit(disp=False, maxiter=200)
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


# %% =========================
# PASO 3: VALIDACIÓN DE SUPUESTOS
# ============================

# Grilla: residuales, FAC residuales y FAC residuales² para cada modelo
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for i, nombre in enumerate(nombres_modelos):
    resultado = estimaciones[nombre]
    p = resultado.model.order[0]
    q = resultado.model.order[2]
    n_inicial = max(p, q, 1)

    residuos          = resultado.resid.dropna().iloc[n_inicial:]
    residuos_cuadrado = residuos**2

    # Gráfica de residuales
    axes[i, 0].plot(residuos, color="black", linewidth=1)
    axes[i, 0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[i, 0].set_title(f"Residuales {nombre}")
    axes[i, 0].set_xlabel("Fecha")
    axes[i, 0].set_ylabel("Residuales")

    # FAC de los residuales (verifica no autocorrelación)
    plot_acf(
        residuos,
        lags=15,
        alpha=0.05,
        bartlett_confint=False,
        ax=axes[i, 1]
    )
    axes[i, 1].set_title(f"FAC residuales {nombre}")
    axes[i, 1].set_ylim(-1, 1)
    axes[i, 1].set_xlabel("Rezago")
    axes[i, 1].set_ylabel("ACF")

    # FAC de los residuales al cuadrado (verifica homocedasticidad)
    plot_acf(
        residuos_cuadrado,
        lags=15,
        alpha=0.05,
        bartlett_confint=False,
        ax=axes[i, 2]
    )
    axes[i, 2].set_title(f"FAC residuales² {nombre}")
    axes[i, 2].set_ylim(-1, 1)
    axes[i, 2].set_xlabel("Rezago")
    axes[i, 2].set_ylabel("ACF")

plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "03_validacion_residuales.png", dpi=150)
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
# %%
