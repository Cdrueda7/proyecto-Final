# Importar las bibliotecas necesarias
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Cargar el dataset
boston_df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv')

# Ver las primeras filas del dataset
print(boston_df.head())

# Estadísticas básicas
print(boston_df.describe())

# Comprobar los tipos de datos
print(boston_df.dtypes)

# Comprobar valores faltantes
print(boston_df.isnull().sum())

### 1. Análisis: ¿Hay una diferencia significativa en el valor medio de las casas cercanas al río Charles o no?
# Graficar los valores medianos de las casas cerca del río Charles y no
sns.boxplot(x='CHAS', y='MEDV', data=boston_df)
pyplot.title('Median Housing Values for Houses Near the Charles River vs. Not')
pyplot.xlabel('Near Charles River (1: Yes, 0: No)')
pyplot.ylabel('Median Housing Value (MEDV)')
pyplot.show()

# Realizar prueba t para ver si hay una diferencia significativa en las medianas
charles_river = boston_df[boston_df['CHAS'] == 1]['MEDV']
no_charles_river = boston_df[boston_df['CHAS'] == 0]['MEDV']

# Prueba t
t_stat, p_value = scipy.stats.ttest_ind(charles_river, no_charles_river)
print(f'T-statistic: {t_stat}, P-value: {p_value}')

# Si p-value < 0.05, podemos rechazar la hipótesis nula de que no hay diferencia significativa.

### 2. Análisis: ¿Hay una diferencia en los valores medianos de las casas para cada proporción de unidades ocupadas por propietarios construidas antes de 1940?
# Graficar la relación de AGE con los valores medianos de las casas
sns.boxplot(x='AGE', y='MEDV', data=boston_df)
pyplot.title('Median Housing Values by Proportion of Owner-Occupied Units Built Before 1940')
pyplot.xlabel('Proportion of Houses Built Before 1940')
pyplot.ylabel('Median Housing Value (MEDV)')
pyplot.show()

# Realizar ANOVA para comprobar si hay una diferencia significativa en los valores medianos de las casas según la edad
model = ols('MEDV ~ AGE', data=boston_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Si el p-value en la tabla de ANOVA es menor a 0.05, podemos concluir que hay una diferencia significativa.

### 3. Análisis: ¿Hay una relación entre la concentración de óxidos de nitrógeno (NOX) y la proporción de acres de negocios no minoristas por ciudad?
# Graficar la relación entre NOX e INDUS
sns.scatterplot(x='NOX', y='INDUS', data=boston_df)
pyplot.title('Nitric Oxide Concentration vs. Proportion of Non-Retail Business Acres')
pyplot.xlabel('Nitric Oxide Concentration (NOX)')
pyplot.ylabel('Proportion of Non-Retail Business Acres (INDUS)')
pyplot.show()

# Calcular el coeficiente de correlación para ver la relación entre NOX e INDUS
correlation = boston_df[['NOX', 'INDUS']].corr()
print(correlation)

# Si la correlación es alta (por ejemplo, > 0.7 o < -0.7), puede indicar una relación fuerte.

### 4. Análisis: ¿Cuál es el impacto de una distancia adicional ponderada a los cinco centros de empleo de Boston en el valor medio de las viviendas ocupadas por propietarios?
# Graficar la relación entre la distancia (DIS) y el valor medio de las viviendas (MEDV)
sns.scatterplot(x='DIS', y='MEDV', data=boston_df)
pyplot.title('Impact of Distance to Employment Centers on Housing Prices')
pyplot.xlabel('Weighted Distance to Employment Centers (DIS)')
pyplot.ylabel('Median Housing Value (MEDV)')
pyplot.show()

# Realizar una regresión lineal para modelar el impacto de DIS sobre MEDV
model = ols('MEDV ~ DIS', data=boston_df).fit()
print(model.summary())

# Si el valor de p es menor a 0.05, podemos concluir que la distancia tiene un impacto significativo sobre el valor medio de las viviendas.
