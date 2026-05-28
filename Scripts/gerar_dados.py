import pandas as pd
import numpy as np

# Ler o arquivo original
df = pd.read_csv("../dados/sensores_agricolas_pandas.csv")

# Lista para armazenar novos registros
novos_dados = []

# Gerar registros até chegar em 500 linhas
while len(novos_dados) < 500:
    linha = df.sample(n=1).iloc[0].copy()

    # Nutrientes
    linha['N'] = np.random.randint(0, 2)
    linha['P'] = np.random.randint(0, 2)
    linha['K'] = np.random.randint(0, 2)

    # Temperatura
    linha['temperatura'] = round(
        np.random.uniform(20, 35),
        1
    )

    # Umidade
    linha['umidade'] = round(
        np.random.uniform(20, 90),
        1
    )

    # pH
    linha['ph'] = round(
        np.random.uniform(4.5, 7.5),
        2
    )

    # Chuva
    linha['chuva'] = round(
        np.random.uniform(0, 250),
        1
    )

    # Cultura aleatória
    culturas = [
        'milho',
        'soja',
        'trigo',
        'café'
    ]

    linha['cultura'] = np.random.choice(culturas)

    # Irrigação
    linha['irrigacao'] = np.random.choice([
        'LIGADA',
        'DESLIGADA'
    ])

    # Adiciona coluna de rendimentos simulada
    linha['rendimento'] = round(
        (
            linha['N'] * 20 +
            linha['P'] * 15 +
            linha['K'] * 18 +
            linha['umidade'] * 1.5 +
            linha['chuva'] * 0.3 -
            abs(linha['ph'] - 6.5) * 12 -
            abs(linha['temperatura'] - 25) * 2 +
            np.random.randint(-3, 3)
        ),
        2
    )

    novos_dados.append(linha)

# Criar um novo data frame
novo_df = pd.DataFrame(novos_dados)

# Salvar novo arquivo
novo_df.to_csv(
    "../dados/dataset_agricola.csv",
    index=False
)

print("Dataset criado com sucesso!")

print(novo_df.head())