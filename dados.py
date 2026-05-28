import pandas as pd
from sklearn.model_selection import train_test_split

MATRICULA = 2025002299 
def gerar_amostra():
    caminho_original = "dados_log.csv/PS_20174392719_1491204439457_log.csv"
    
    df = pd.read_csv(caminho_original)
    
    fraudes = df[df["isFraud"] == 1].sample(n=500, random_state=MATRICULA)
    normais = df[df["isFraud"] == 0].sample(n=2500, random_state=MATRICULA)
    
    amostra = pd.concat([fraudes, normais])
    amostra = amostra.sample(frac=1, random_state=MATRICULA).reset_index(drop=True)
    
    amostra.to_csv("paysim_sample.csv", index=False)
    print("Sucesso: Arquivo 'paysim_sample.csv' gerado com 3000 instâncias!")

def carregar_dados():
    df = pd.read_csv("paysim_sample.csv")
    
    X = df.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'])
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.20, 
        stratify=y, 
        random_state=MATRICULA 
    )
    
    return X_train, X_test, y_train, y_test
