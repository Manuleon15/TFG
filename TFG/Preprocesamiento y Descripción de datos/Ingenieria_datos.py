# Se importan las librerías necesarias
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns
import numpy as np

# Elección de los 21 ETF's
tickers = ['TLT', 'SHY', 'IEF',    # Bonos del Estado
           'VNQ', 'REZ', 'IYR',   # Real Estate
           'VAW', 'FXZ', 'FMAT',   # Materiales
           'IXJ', 'IHI', 'FBT',   # Sanidad
           'TAN', 'ICLN', 'QCLN',  # Energías Alternativas
           'VGT', 'XLK', 'IYW',    # Tecnología
           'XLF', 'EUFN', 'FXO']   # Finanzas

# Último día para obtener datos (8 de marzo de 2024)
end_date = datetime(datetime.today().year, 3, 8)
print(end_date)

# Primer día de obtención de datos (10 años atrás)
start_date = end_date - timedelta(days=10*365)
print(start_date)


# Se obtienen los datos de la Api de Yahoo Finance
close_df = pd.DataFrame()


# se obtiene el precio de cierre ajustado
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    close_df[ticker] = data['Adj Close']

print(close_df)

# Verificamos el tipo de las variables
tipo = close_df.dtypes
print(tipo)

# Verificar valores nulos en la base de datos
null_values = close_df.isnull()

# Mostrar el conteo de valores nulos por columna
print("Conteo de valores nulos por columna:")
print(null_values.sum())


# Se guarda la base de datos como csv
close_df.to_csv('close_df.csv', index=True)

# ETL
import matplotlib
matplotlib.use('TkAgg')   # Backend para que funcione matplotlib
import matplotlib.pyplot as plt


"""Dispersión"""
# Símbolos de los ETFs según el sector y del S&P 500 (se repite para cada sector)
etf_symbols = ['XLF', 'EUFN', 'FXO']
sp500_symbol = '^GSPC'

# Descargar datos históricos de cierre ajustado del 2010 al 2022
etf_data = yf.download(etf_symbols, start=start_date, end=end_date)['Adj Close']
sp500_data = yf.download(sp500_symbol, start=start_date, end=end_date)['Adj Close']

# Calcular la media de los rendimientos diarios para los ETFs de bonos
finance_etfs_returns = etf_data.pct_change().mean(axis=1)

# Crear un DataFrame combinado con los datos de los rendimientos y del S&P 500
combined_data = pd.DataFrame({'Media de retornos de los ETFs de Finanzas': finance_etfs_returns,
                              'S&P 500': sp500_data.pct_change()}).dropna()

# Crear un gráfico de dispersión
sns.scatterplot(x='S&P 500', y='Media de retornos de los ETFs de Finanzas', data=combined_data)

# Añadir título y etiquetas
plt.title('Relación entre los Rendimientos de ETFs de Finanzas y S&P 500')
plt.xlabel('Retornos S&P 500')
plt.ylabel('Retornos ETFs de Finanzas')

# Mostrar el gráfico
plt.show()


"""---------------Modificaciones de la BBDD para modelos ARIMA, ARIMAX, y GARCH-----------------------"""
# Se carga el csv con la base de datos
# Se indica que la columna ínidice va a ser la columna de Fecha, y que se trata de datos de fecha
df = pd.read_csv('/close_df.csv', index_col='Date', parse_dates=True)


# Se seleccionan los 3 activos que conformen el sector a estudiar
sector_tecnologia = df.iloc[:, 15:18]

# Se calcula la media de sus precios para tener un único precio del sector
precio_sector = sector_tecnologia.mean(axis=1).dropna()

# Se guarda el csv
precio_sector.to_csv('precios_sector_tecnologia.csv', index=True)

# Este proceso se repite para los 7 sectores
