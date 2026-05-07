# PROYECTO: Código de Google Colab
# TIPO DE PROBLEMA: CLASIFICACIÓN BINARIA
# VARIABLE DEPENDIENTE: churn


# 1. IMPORTAR LIBRERÍAS
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Modelos de regresión
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Métricas
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score

# Preprocesamiento
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Visualización
import seaborn as sns
import matplotlib.pyplot as plt


# 2. CARGAR DATASET
dataframe = pd.read_csv("/content/Bank Customer Churn Prediction.csv")

#print("Primeras filas del dataset:")
#print(dataframe.head())

#print("\nColumnas del dataset:")
#print(dataframe.columns)

#print("\nTamaño del dataset:")
#print(dataframe.shape)


# 3. SELECCIONAR LAS COLUMNAS QUE SE VAN A USAR
df = dataframe[[
    "credit_score",
    "country",
    "gender",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary",
    "churn"
]].copy()


# 4. LIMPIEZA BÁSICA DE DATOS
#print("\nValores nulos antes de limpiar:")
#print(df.isnull().sum())

#print("\nDuplicados antes de limpiar:")
#print(df.duplicated().sum())

df = df.dropna()
df = df.drop_duplicates()

#print("\nValores nulos después de limpiar:")
#print(df.isnull().sum())

#print("\nDuplicados después de limpiar:")
#print(df.duplicated().sum())

#print("\nTamaño del dataset después de limpiar:")
#print(df.shape)


# 5. CODIFICAR VARIABLES CATEGÓRICAS
le_country = LabelEncoder()
le_gender = LabelEncoder()

df["country"] = le_country.fit_transform(df["country"])
df["gender"] = le_gender.fit_transform(df["gender"])

print("\nDataset después de codificar country y gender:")
print(df.head())


# REGRESIÓN LINEAL SIMPLE
print("REGRESIÓN LINEAL SIMPLE")

df_churn_edad = df.groupby("age", as_index=False)["churn"].mean()
df_churn_edad = df_churn_edad.rename(columns={"churn": "tasa_churn"})

print("\nTasa promedio de churn por edad:")
print(df_churn_edad.head())

valores_x = df_churn_edad[["age"]].values
valores_y = df_churn_edad["tasa_churn"].values

regresion_lineal_simple = LinearRegression().fit(valores_x, valores_y)

print(f"Intercepto B0 -> {regresion_lineal_simple.intercept_:.4f}")
print(f"Coeficiente B1 -> {regresion_lineal_simple.coef_[0]:.4f}")

y_pred_simple = regresion_lineal_simple.predict(valores_x)

mse_simple = mean_squared_error(valores_y, y_pred_simple)
rmse_simple = np.sqrt(mse_simple)
r2_simple = r2_score(valores_y, y_pred_simple)

print(f"MSE = {mse_simple:.4f}")
print(f"RMSE = {rmse_simple:.4f}")
print(f"R2 = {r2_simple:.4f}")

X_plot = np.linspace(valores_x.min(), valores_x.max(), 200).reshape(-1, 1)

plt.figure()
plt.scatter(valores_x, valores_y)
plt.plot(X_plot, regresion_lineal_simple.predict(X_plot), color="red")
plt.xlabel("Age")
plt.ylabel("Tasa promedio de churn")
plt.title("Regresión Lineal Simple: Age vs Tasa de Churn")
plt.show()


# REGRESIÓN LINEAL MÚLTIPLE
print("\n\n")
print("REGRESIÓN LINEAL MÚLTIPLE")
valores_x_multiple = df[[
    "credit_score",
    "country",
    "gender",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary"
]].values

valores_y_multiple = df["churn"].values

regresion_lineal_multiple = LinearRegression().fit(
    valores_x_multiple,
    valores_y_multiple
)

print(f"B0 = {regresion_lineal_multiple.intercept_:.4f}")

print(f"B1 credit_score = {regresion_lineal_multiple.coef_[0]:.4f}")
print(f"B2 country = {regresion_lineal_multiple.coef_[1]:.4f}")
print(f"B3 gender = {regresion_lineal_multiple.coef_[2]:.4f}")
print(f"B4 age = {regresion_lineal_multiple.coef_[3]:.4f}")
print(f"B5 tenure = {regresion_lineal_multiple.coef_[4]:.4f}")
print(f"B6 balance = {regresion_lineal_multiple.coef_[5]:.8f}")
print(f"B7 products_number = {regresion_lineal_multiple.coef_[6]:.4f}")
print(f"B8 credit_card = {regresion_lineal_multiple.coef_[7]:.4f}")
print(f"B9 active_member = {regresion_lineal_multiple.coef_[8]:.4f}")
print(f"B10 estimated_salary = {regresion_lineal_multiple.coef_[9]:.8f}")

y_pred_multiple = regresion_lineal_multiple.predict(valores_x_multiple)

mse_multiple = mean_squared_error(valores_y_multiple, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(valores_y_multiple, y_pred_multiple)

print(f"MSE = {mse_multiple:.4f}")
print(f"RMSE = {rmse_multiple:.4f}")
print(f"R2 = {r2_multiple:.4f}")


# REGRESIÓN POLINÓMICA

print("\n\n")
print("REGRESIÓN POLINÓMICA")

valores_x_poli = df_churn_edad[["age"]].values
valores_y_poli = df_churn_edad["tasa_churn"].values


# Regresión Polinómica de Grado 2
regresion_polinomica_2 = PolynomialFeatures(degree=2)

valores_x_poli2 = regresion_polinomica_2.fit_transform(valores_x_poli)

regresion_polinomica2 = LinearRegression().fit(
    valores_x_poli2,
    valores_y_poli
)

y_predict_polinomica2 = regresion_polinomica2.predict(valores_x_poli2)

r2_polinomica2 = r2_score(valores_y_poli, y_predict_polinomica2)
mse_polinomica2 = mean_squared_error(valores_y_poli, y_predict_polinomica2)
rmse_polinomica2 = np.sqrt(mse_polinomica2)

plt.figure()
plt.scatter(valores_x_poli, valores_y_poli)
plt.plot(
    X_plot,
    regresion_polinomica2.predict(regresion_polinomica_2.transform(X_plot)),
    color="red"
)
plt.xlabel("Age")
plt.ylabel("Tasa promedio de churn")
plt.title("Regresión Polinómica de Grado 2")
plt.show()

print(f"--- Regresión Polinómica Grado 2 ---")
print(f"R2 = {r2_polinomica2:.4f}, MSE = {mse_polinomica2:.4f}, RMSE = {rmse_polinomica2:.4f}")


# Regresión Polinómica de Grado 3
regresion_polinomica_3 = PolynomialFeatures(degree=3)

valores_x_poli3 = regresion_polinomica_3.fit_transform(valores_x_poli)

regresion_polinomica3 = LinearRegression().fit(
    valores_x_poli3,
    valores_y_poli
)

y_predict_polinomica3 = regresion_polinomica3.predict(valores_x_poli3)

r2_polinomica3 = r2_score(valores_y_poli, y_predict_polinomica3)
mse_polinomica3 = mean_squared_error(valores_y_poli, y_predict_polinomica3)
rmse_polinomica3 = np.sqrt(mse_polinomica3)

plt.figure()
plt.scatter(valores_x_poli, valores_y_poli)
plt.plot(
    X_plot,
    regresion_polinomica3.predict(regresion_polinomica_3.transform(X_plot)),
    color="red"
)
plt.xlabel("Age")
plt.ylabel("Tasa promedio de churn")
plt.title("Regresión Polinómica de Grado 3")
plt.show()

print(f"--- Regresión Polinómica Grado 3 ---")
print(f"R2 = {r2_polinomica3:.4f}, MSE = {mse_polinomica3:.4f}, RMSE = {rmse_polinomica3:.4f}")


# CLASIFICACIÓN

# 6. SEPARAR VARIABLES INDEPENDIENTES Y VARIABLE DEPENDIENTE
X = df[[
    "credit_score",
    "country",
    "gender",
    "age",
    "tenure",
    "balance",
    "products_number",
    "credit_card",
    "active_member",
    "estimated_salary"
]]

y = df["churn"]


# 7. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

print("\nTamaño de X_train:", X_train.shape)
print("Tamaño de X_test:", X_test.shape)
print("Tamaño de y_train:", y_train.shape)
print("Tamaño de y_test:", y_test.shape)


# 8. ESCALAR DATOS PARA REGRESIÓN LOGÍSTICA
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 9. MODELO 1: LOGISTIC REGRESSION
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

print("\n\n")
print("\nLOGISTIC REGRESSION")
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_log))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_log))

accuracy_logistic = accuracy_score(y_test, y_pred_log)

print("Accuracy Logistic Regression:", accuracy_logistic)


# 10. MODELO 2: RANDOM FOREST
rf_model = RandomForestClassifier(n_estimators=100, random_state=101)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n\n")
print("\nRANDOM FOREST")
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_rf))

accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("Accuracy Random Forest:", accuracy_rf)


# 11. VISUALIZACIÓN DE MATRIZ DE CONFUSIÓN - RANDOM FOREST
cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión - Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()


# 12. COMPARACIÓN FINAL DE MODELOS
print("\n\n")
print("\nACCURACY")
print("Accuracy Logistic Regression:", accuracy_logistic)
print("Accuracy Random Forest:", accuracy_rf)

if accuracy_rf > accuracy_logistic:
    print("\nEl modelo con mejor accuracy fue Random Forest.")
elif accuracy_logistic > accuracy_rf:
    print("\nEl modelo con mejor accuracy fue Logistic Regression.")
else:
    print("\nAmbos modelos tuvieron el mismo accuracy.")