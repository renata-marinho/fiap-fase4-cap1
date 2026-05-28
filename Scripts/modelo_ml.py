import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error,r2_score)
import joblib

# ===================================
# 1. Ler os dados
# ===================================

df = pd.read_csv("../dados/dataset_agricola.csv")
print(df.head())

# ===================================
# 2. Definir entradas (X)
# ===================================

X = df[['N', 'P', 'K', 'temperatura', 'umidade', 'ph', 'chuva']]

# ===================================
# 3. Definir saída (y)
# ===================================

y = df['rendimento']

# ===================================
# 4. Separar treino e teste
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===================================
# 5. Criar modelos
# ===================================

modelos = {
    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42
    ),
    "Random Forest Regressor": RandomForestRegressor(
    n_estimators=100,
    random_state=42
    )
}

# ===================================
# 6. Treinar modelos
# ===================================

resultados = []

for nome_modelo, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    previsoes = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, previsoes)
    mse = mean_squared_error(y_test, previsoes)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, previsoes)

    resultados.append({
        "Modelo": nome_modelo,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

# ===================================
# 7. Mostrar resultados
# ===================================

df_resultados = pd.DataFrame(resultados)

print("\n===== COMPARAÇÃO DOS MODELOS =====")
print(df_resultados)

# ===================================
# 8. Escolher melhor modelo
# ===================================

melhor_resultado = df_resultados.sort_values(by="R2", ascending=False).iloc[0]
melhor_nome = melhor_resultado["Modelo"]
melhor_modelo = modelos[melhor_nome]

print("\n===== MELHOR MODELO =====")
print(f"Melhor modelo: {melhor_nome}")
print(f"R²: {melhor_resultado['R2']:.2f}")
print(f"RMSE: {melhor_resultado['RMSE']:.2f}")

# ===================================
# 9. Salvar melhor modelo
# ===================================

joblib.dump(melhor_modelo, "modelo.pkl")
print("\nMelhor modelo salvo com sucesso!")