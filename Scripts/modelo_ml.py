import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
# 5. Criar modelo
# ===================================

modelo = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# ===================================
# 6. Treinar modelo
# ===================================

modelo.fit(X_train, y_train)

# ===================================
# 7. Fazer previsões
# ===================================

previsoes = modelo.predict(X_test)

# ===================================
# 8. Avaliar métricas
# ===================================

mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)
rmse = mse ** 0.5
r2 = r2_score(y_test, previsoes)

# ===================================
# 9. Mostrar resultados
# ===================================

print("\n===== MÉTRICAS =====")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# ===================================
# 10. Salvar modelo
# ===================================

joblib.dump(modelo, "modelo.pkl")
print("\nModelo salvo com sucesso!")