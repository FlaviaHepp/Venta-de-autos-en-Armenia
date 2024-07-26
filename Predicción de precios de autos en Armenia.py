"""Este conjunto de datos contiene información sobre las ventas de automóviles en Armenia, necesita predecir el precio del automóvil, datos 
tomados del sitio web list.am.

A continuación se muestra la descripción de las columnas:

*Nombre del coche: El nombre o modelo del coche.

*Año: El año de fabricación del automóvil.

*Región: El área geográfica donde se vende o se registró el automóvil.

*Tipo de combustible: el tipo de combustible que utiliza el automóvil.

*Kilometraje: La distancia total que ha recorrido el automóvil, normalmente medida en millas o kilómetros.

*Precio: El precio de venta del coche, normalmente en la moneda local."""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('Armenian Market Car Prices.csv')
df_g = df.copy()
df.info()

df.drop(columns=["Car Name"], axis = 1, inplace=True)
le = LabelEncoder()
df["FuelType"] = le.fit_transform(df["FuelType"])
df["Region"] = le.fit_transform(df["Region"])
corr = df.corr()
sns.heatmap(corr, annot=True)

y = df["Price"]
X = df.drop(columns=["Price"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor()
xgb = XGBRegressor()
lr = LinearRegression()

for model in [rf, xgb, lr]:
    model.fit(X_train, y_train)
    print(model.__class__.__name__, model.score(X_test, y_test))

ridge = Ridge(alpha = 0.1)
ridge.fit(X_train, y_train)
print(ridge.score(X_test, y_test))

lasso = Lasso(alpha = 0.1)
lasso.fit(X_train, y_train)
print(lasso.score(X_test, y_test))

elastic = ElasticNet(alpha = 0.1)
elastic.fit(X_train, y_train)
print(elastic.score(X_test, y_test))

sns.pairplot(df_g)

plt.figure(figsize=(15, 10))
fuel_type = df_g["FuelType"].value_counts()
sns.barplot(x=fuel_type.values, y=fuel_type.index)
plt.show()

plt.figure(figsize=(15, 10))
region = df_g["Region"].value_counts()
total= df_g["Region"].count()
border = total * 0.02

region_use = region[region > border]
region_use["Other"] = region[region <= border].sum()

color = sns.color_palette("pastel")[0:18]
explode = [0.2 if i == 0 else 0 for i in range(len(region_use))]

plt.pie(region_use.values, labels=region_use.index, autopct='%1.0f%%'
,  colors = color, shadow = True, startangle = 145,wedgeprops={'edgecolor': 'black'},explode=explode)

plt.title("Distribución de regiones\n", fontsize = '16', fontweight = 'bold')
plt.show()

# Comprobar valores faltantes
print(df.isnull().sum())

# Convierta variables categóricas a numéricas usando la codificación de etiquetas
le = LabelEncoder()
#df['Car Name'] = le.fit_transform(df['Car Name'])
df['Region'] = le.fit_transform(df['Region'])
df['FuelType'] = le.fit_transform(df['FuelType'])

# Análisis de datos exploratorios (EDA)
# Visualizar la distribución de la variable objetivo (Precio)
sns.histplot(df['Price'], kde=True)
plt.title('Distribución de precios\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Analizar la relación entre las características y la variable objetivo
sns.pairplot(df)
plt.show()

# Selección del modelo
# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrene y evalúe modelos mediante validación cruzada
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'{name} Actuación:')
    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
    print(f'MSE: {mean_squared_error(y_test, y_pred)}')
    print(f'R-cuadrado: {r2_score(y_test, y_pred)}\n')

# Ajuste de hiperparámetros (ejemplo de bosque aleatorio)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

print('Mejores parámetros:', grid_search.best_params_)

# Entrenamiento y predicción del modelo final
final_model = RandomForestRegressor(n_estimators=200, max_depth=30)
final_model.fit(X_train, y_train)
final_predictions = final_model.predict(X_test)

# Evaluar el rendimiento del modelo final
print('Rendimiento final del modelo:')
print(f'MAE: {mean_absolute_error(y_test, final_predictions)}')
print(f'MSE: {mean_squared_error(y_test, final_predictions)}')
print(f'R-cuadrado: {r2_score(y_test, final_predictions)}')

# Obtener información a partir de los datos
data = pd.read_csv('Armenian Market Car Prices.csv')
print(data.head())

print(data.describe())

#Porcentaje de valores faltantes en el conjunto de datos
missing_percentage = (data.isnull().sum() / len(data)) * 100
print(missing_percentage)

data.columns

label_encoder = LabelEncoder()

# Codificar columnas categóricas
categorical_cols = ['Car Name','Region', 'FuelType']

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Imprima las primeras filas para comprobar
print(data.head())

print(data.head(10))

# Calcular la edad de los coches
data['Car Age'] = 2024 - data['Year']

# Distribución parcelaria de edades de los coches
plt.figure(figsize=(10, 6))
sns.histplot(data['Car Age'], kde=True, bins=10, color = "cyan", edgecolor = "blue")
plt.title('Distribución de edades de los automóviles\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Edad del coche\n')
plt.ylabel('Frecuencia\n')
plt.show()

# Distribución parcelaria de los precios de los coches
plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], kde=True, bins=10, color = "fuchsia", edgecolor = "white")
plt.title('Distribución de precios de automóviles\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Precio\n')
plt.ylabel('Frecuencia\n')
plt.show()

# Diagrama de dispersión de kilometraje frente a precio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Mileage', y='Price', data=data)
plt.title('Kilometraje versus precio\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Kilometraje\n')
plt.ylabel('Precio\n')
plt.show()

# Trazar la distribución de los tipos de combustible
plt.figure(figsize=(10, 6))
sns.countplot(x='FuelType', data=data)
plt.title('Distribución de tipos de combustible\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Tipo de combustible\n')
plt.ylabel('Conteo\n')
plt.show()

# Mapa de calor de correlación
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='inferno', linewidths=0.5)
plt.title('Mapa de calor de correlación\n', fontsize = '16', fontweight = 'bold')
plt.show()

# Contar parcela de autos por año
plt.figure(figsize=(14, 8))
sns.countplot(x='Year', data=data)
plt.title('Recuento de vehículos por año\n', fontsize = '16', fontweight = 'bold')
plt.xlabel('Año\n')
plt.ylabel('Conteo\n')
plt.xticks(rotation=90)
plt.show()

