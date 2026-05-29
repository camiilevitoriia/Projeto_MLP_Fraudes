import dados
import treino
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def executar_experimentos_finais():
    print("A iniciar Bateria Final: Análise de Fatores, Tabela e Boxplot...")
    
    X_train, X_test, y_train, y_test = dados.carregar_dados()
    
    configuracoes = [
        {"nome": "Padrão (10 Neurónios, Drop 0.3)", "neur": 10, "drop": 0.3},
        {"nome": "Fator 1: Alta Capac. (50 Neurónios)", "neur": 50, "drop": 0.3},
        {"nome": "Fator 2: Sem Reg. (Drop 0.0)", "neur": 10, "drop": 0.0}
    ]
    
    repeticoes = 10
    resultados_completos = [] 
    
    for config in configuracoes:
        print(f"\n{'='*60}")
        print(f"TESTANDO CONFIGURAÇÃO: {config['nome']}")
        print(f"{'='*60}")
        
        for i in range(repeticoes):
            print(f"\nRepetição {i+1}/{repeticoes}...")
            inicio = time.time()
            
            modelo, pipeline = treino.treinar_modelo_kfold(
                X_train=X_train, 
                y_train=y_train, 
                batch_size=32, 
                lr=0.01, 
                max_epochs=50, 
                patience=5,
                num_neuronios=config["neur"],
                taxa_dropout=config["drop"]
            )
            
            X_test_prep = pipeline.transform(X_test)
            X_t_test = torch.tensor(X_test_prep, dtype=torch.float32)
            
            modelo.eval()
            with torch.no_grad():
                previsoes = (modelo(X_t_test) >= 0.5).float().cpu().numpy()
                
            fim = time.time()
            tempo_gasto = fim - inicio
            
            acc = accuracy_score(y_test, previsoes)
            prec = precision_score(y_test, previsoes, zero_division=0)
            rec = recall_score(y_test, previsoes, zero_division=0)
            f1 = f1_score(y_test, previsoes, zero_division=0)
            
            resultados_completos.append({
                "Configuração": config['nome'],
                "Repetição": i + 1,
                "Acurácia": round(acc, 4),
                "Precisão": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "Tempo (s)": round(tempo_gasto, 2)
            })
            print(f"F1-Score obtido: {f1:.4f}")

    df_resultados = pd.DataFrame(resultados_completos)
    df_resultados.to_csv("resultados/tabela_metricas_completa.csv", index=False)

    df_f1 = df_resultados[["Configuração", "Repetição", "F1-Score"]]
    df_f1.to_csv("resultados/resultados_experimentos.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Configuração", y="F1-Score", data=df_resultados)
    plt.title("Comparação de Desempenho (F1-Score) - Análise de Fatores")
    plt.ylabel("F1-Score (Maior é melhor)")
    plt.xlabel("Configurações da Rede Neural")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("resultados/boxplot_comparacao.png")
    plt.close()
    
    medias = df_resultados.groupby("Configuração").mean().drop(columns=["Repetição"])
    print("\n" + "="*60)
    print("MÉDIAS FINAIS DOS EXPERIMENTOS")
    print("="*60)
    print(medias)
    
    print("\nConcluído! Ficheiro 'tabela_metricas_completa.csv' gerado.")
    print("Imagem 'curvas_aprendizado.png' gerada (pelo treino.py).")
    print(" Imagem 'boxplot_comparacao.png' gerada com sucesso!")

if __name__ == "__main__":
    executar_experimentos_finais()
