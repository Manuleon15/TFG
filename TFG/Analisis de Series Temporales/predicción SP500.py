import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# La predicción del SP500 será necesaria para utilizarla como input en las predicciones del modelo ARIMAX

ticker = 'SPY'

# Descargamos los datos para el SP500 para las fechas indicadas
data = yf.download(ticker, start="2014-03-14", end="2024-03-08").dropna()

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


ad_test(data['Adj Close'])
# Teniendo un P-Value >0.05, podemos aceptar la hipótesis nula de que el dataframe no tiene estacionariedad
# Por ello, d=1 al crear el modelo ARIMA


fit = auto_arima(data['Adj Close'], trace=True, suppress_warnings=True,
                 stepwise=True, information_criterion='aic')


# Separamos los datos en train y test con un 80% para train y un 20% para test
# Cogemos de la observación primera hasta el 80% de los datos para train
# Y del 80% hasta el final (20%) para test
train_data, test_data = data['Adj Close'][0:int(len(data)*0.8)], data['Adj Close'][int(len(data)*0.8):]

# Train y test para el conjunto de datos
train_arima = train_data
test_arima = test_data

# Para los datos de train los vamos asignando a la variable history
history = [x for x in train_arima]
y = test_arima
# Hacemos la primera predicción con los parámetros óptimos y la asignamos a la variable predictions
predictions = list()
model = ARIMA(history, order=(2, 1, 2))
model_fit = model.fit()
yhat = model_fit.forecast()[0]
predictions.append(yhat)
history.append(y[0])

# Predicciones siguientes
for i in range(1, len(y)):
    # predecimos al igual que el proceso anterior
    model = ARIMA(history, order=(2, 1, 2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # los asignamos a la variable predictions
    predictions.append(yhat)
    # la observación se asigna a la variable history
    obs = y[i]
    history.append(obs)

# Predecimos el día siguiente del cual no tenemos datos
n_steps = 1
forecast = model_fit.forecast(steps=n_steps)
print(forecast)

