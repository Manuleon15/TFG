# Importamos las librerías necesarias
import warnings

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
import statsmodels.api as sm
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

# Leemos la base de datos con los precios diarios del sector Materiales
df = pd.read_csv('/Users/manuel/Desktop/pythonProject/precios_sector_materiales.csv', index_col='Date',
                 parse_dates=True)


# Creamos la función que nos ayuda a determinar la estacionariedad de los datos
def ad_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Número de lags : ", dftest[2])
    print("4. Número de observaciones usadas por la regresión ADF y cálculo de valores críticos : ", dftest[3])
    print("5. Valores Críticos : ")
    for key, val in dftest[4].items():
        print("\t", key, ". ", val)


ad_test(df['0'])
# Teniendo un P-Value >0.05, no podemos rechazar la hipótesis nula
# Esto significa que no se puede afirmar que la serie sea estacionaria

# Con la función auto_arima obtenemos los parámetros óptimos para el modelo
fit = auto_arima(df, trace=True, suppress_warnings=True,
                 stepwise=False, information_criterion='aic')

# Separamos los datos en train y test con un 80% para train y un 20% para test
# Cogemos de la observación primera hasta el 80% de los datos para train
# Y del 80% hasta el final (20%) para test
train_data, test_data = df[0:int(len(df) * 0.8)], df[int(len(df) * 0.8):]

# Train y test para el conjunto de datos
train_arima = train_data['0']
test_arima = test_data['0']

time_series = df['0']

# Aplicamos la descomposición temporal
result = seasonal_decompose(time_series, model='additive', period=12)
seasonal = result.seasonal
trend = result.trend
residual = result.resid

# Gráfico de la descomposición de la serie temporal
plt.figure(figsize=(14, 7))
plt.subplot(411)
plt.plot(df['0'], label='Serie Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Tendencia')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Estacionalidad')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuos')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Gráfico train y test
plt.figure(figsize=(10, 5))
plt.plot(train_data['0'], label='Datos Entrenamiento')
plt.plot(test_data['0'], label='Datos Test')
plt.title('Datos Entrenamiento y Test Sector Materiales')
plt.legend()
plt.show()

# Para los datos de train los vamos asignando a la variable history
history = [x for x in train_arima]
y = test_arima
# Hacemos la primera predicción con los parámetros óptimos y la asignamos a la variable predictions
predictions = list()
model = ARIMA(history, order=(4, 1, 1))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])

# Extraemos los residuos - Se usarán para el modelo GARCH
residuals = model_fit.resid

# Predicciones siguientes
for i in range(1, len(y)):
    # predecimos al igual que el proceso anterior
    model = ARIMA(history, order=(4, 1, 1))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # los asignamos a la variable predictions
    predictions.append(yhat)
    # la observación se asigna a la variable history
    obs = y[i]
    history.append(obs)

# Calculamos el performance del modelo
mse = mean_squared_error(y, predictions)
print('MSE: ' + str(mse))
mae = mean_absolute_error(y, predictions)
print('MAE: ' + str(mae))
rmse = math.sqrt(mean_squared_error(y, predictions))
print('RMSE: ' + str(rmse))

# Visualizamos las predicciones
plt.figure(figsize=(16, 8))
plt.plot(df.index[-600:], df['0'].tail(600), color='green', label='Precios Entrenamiento')
plt.plot(test_data.index, y, color='blue', label='Precios reales')
plt.plot(test_data.index, predictions, color='red', label='Predicción precios')
plt.title('Sector Materiales Predicción Precios')
plt.xlabel('Fecha')
plt.ylabel('Precio Sector Materiales')
plt.legend()
plt.grid(True)
plt.show()

# Predecimos el día siguiente del cual no tenemos datos
n_steps = 1
forecast = model_fit.forecast(steps=n_steps)
print(forecast)

"""-------------------------------ARIMAX MODEL------------------------------------------"""

# Para el modelo ARIMAX, necesitamos una variable exógena: en nuestro caso utilizaremos los precios
# del índice SP500, teniendo en cuenta la correlación con el sector.

ticker = 'SPY'

# Descargamos los datos para el SP500 para las fechas indicadas
data = yf.download(ticker, start="2014-03-14", end="2024-03-08").dropna()

# Juntamos ambas df, creando un único df con las columnas de los precios del sector y los del SP500
# eliminando los na.
df = df.merge(data['Adj Close'], left_index=True, right_index=True, how='left').dropna()


# Hacemos el test de aumented fuller
def ad_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Número de lags : ", dftest[2])
    print("4. Número de observaciones usadas por la regresión ADF y cálculo de valores críticos : ", dftest[3])
    print("5. Valores Críticos : ")
    for key, val in dftest[4].items():
        print("\t", key, ". ", val)


ad_test(df['0'])
ad_test(df['Adj Close'])
# Ninguno tiene estacionariedad

# Separamos en train y test
train_data, test_data = df[0:int(len(df) * 0.8)], df[int(len(df) * 0.8):]

# Esta vez también con la variable exógena
train_arima = train_data['0']
test_arima = test_data['0']
train_exog = train_data['Adj Close']
test_exog = test_data['Adj Close']

history = [x for x in train_arima]
exog_history = [x for x in train_exog]
y = test_arima
predictions = []

# Modelo ARIMAX, aunque en python se utiliza aparentemente SARIMAX, no utilizamos la S de "Seasonality" ya que nuestros
# datos no son estacionales como se ha podido observar en los gráficos ya mostrados.
model = SARIMAX(history, order=(4, 1, 1), exog=exog_history)
model_fit = model.fit()

# Predicciones
for i in range(len(y)):
    exog = [test_exog.iloc[i]]  # Variable exógena para esta observación
    model = SARIMAX(history, order=(4, 1, 1), exog=exog_history)
    model_fit = model.fit()
    yhat = model_fit.forecast(exog=[exog])[0]
    predictions.append(yhat)
    # Actualizamos la variable history para el siguiente loop
    history.append(y.iloc[i])
    exog_history.append(exog[0])

# Evaluamos el performance del modelo ARIMAX
mse = mean_squared_error(y, predictions)
print('MSE:', mse)
mae = mean_absolute_error(y, predictions)
print('MAE:', mae)
rmse = math.sqrt(mse)
print('RMSE:', rmse)

# Visualizamos las predicciones
plt.figure(figsize=(16, 8))
plt.plot(df.index[-600:], df['0'].tail(600), color='green', label='Precios Entrenamiento')
plt.plot(test_data.index, y, color='blue', label='Precios reales')
plt.plot(test_data.index, predictions, color='red', label='Predicción precios')
plt.title('Sector Materiales Predicción Precios')
plt.xlabel('Fecha')
plt.ylabel('Precio Sector Materiales')
plt.legend()
plt.grid(True)
plt.show()

# Predicciones futuras con ARIMAX necesita el valor futuro de la variable exógena también
n_steps = 1  # Predicción para el día siguiente del cual no tenemos datos
future_exog = 509.5209948  # Predicción del Sp500 para el día siguiente ya obtenida con anterioridad

forecast = model_fit.forecast(steps=n_steps, exog=future_exog)
print(forecast)

"""------------------------------------GARCH MODEL------------------------------------------"""


# Definimos la función que realice el gráfico de diagnóstico
def plot_diagnostics(residuals):
    # Gráfica Q-Q
    plt.figure(figsize=(6, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title('Gráfica Q-Q')
    plt.ylim([-10, 10])
    plt.show()


# Llamamos a la función con los residuos obtenidos en ARIMA
plot_diagnostics(residuals)

# Hacemos el test de ARCH
p_value = het_arch(residuals)[1]
print(f"Test ARCH, p-value: {p_value}")

# Inicializamos las variables
best_aic = np.inf
best_bic = np.inf
best_model = None
best_params = (0, 0)

# Selección del Modelo
for p in range(1, 4):
    for q in range(1, 4):
        try:
            model = arch_model(residuals, vol='GARCH', p=p, q=q)
            model_fit = model.fit(disp='off')
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_bic = model_fit.bic
                best_model = model_fit
                best_params = (p, q)
        except ValueError as e:
            print(f"Error Garch({p},{q}):", e)

print(f"Mejor Modelo: p={best_params[0]}, q={best_params[1]}, AIC={best_aic}, BIC={best_bic}")

# Realizamos el modelo ljungbox
# Un número de lags comúnmente utilizado para pruebas de autocorrelación es min(10, n/5)
# donde n es el número de observaciones (2011). Esto daría aproximadamente 40
lb_results = acorr_ljungbox(residuals, lags=[40], return_df=True)
print(lb_results)
# no hay evidencia suficiente para rechazar la hipótesis nula
# los residuos parecen comportarse como ruido blanco hasta el retraso 100.

# Predecimos la volatilidad del día siguiente con el mejor modelo
forecast = best_model.forecast(horizon=1)
next_day_volatility = forecast.variance.iloc[-1] ** 0.5
print("Volatilidad predicha:", next_day_volatility.values)

volatility = best_model.conditional_volatility

# Graficamos los residuos y la volatilidad
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.plot(best_model.resid, color='blue', label='Residuos')
plt.title('Residuos Reales')
plt.legend()

plt.subplot(212)
plt.plot(best_model.conditional_volatility, color='red', label='Volatilidad condicional')
plt.title('Volatilidad condicional del modelo GARCH')
plt.legend()

plt.tight_layout()
plt.show()
