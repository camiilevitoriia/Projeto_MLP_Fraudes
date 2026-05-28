import dados
import treino
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def executar_experimentos():
    print("A iniciar a Bateriade Experimentos Científicos...")
    
    X_train, X_test, y_train, y_test = dados.carregar_dados()
    tamanho_total = len(X_train)
    
    configuracoes = {
        "SGD (Estocastico)": 1,
        "Mini-Batch (32)": 32,
        "Mini-Batch (128)": 128,
        "Batch (Completo)": tamanho_total
    }
    
    repeticoes = 10
    resultados_f1 = [] 
    
    for nome, batch_size in configuracoes.items():
        print(f"\n{'='*50}")
        print(f"TESTANDO CONFIGURAÇÃO: {nome}")
        print(f"{'='*50}")
        
        for i in range(repeticoes):
            print(f"\n--- Repetição {i+1}/{repeticoes} para {nome} ---")
            
         
            modelo, pipeline = treino.treinar_modelo_kfold(
                X_train=X_train, 
                y_train=y_train, 
                batch_size=batch_size, 
                lr=0.01, 
                max_epochs=50,
                patience=5
            )
            
            import torch
            from sklearn.metrics import f1_score
            
            X_test_prep = pipeline.transform(X_test)
            X_t_test = torch.tensor(X_test_prep, dtype=torch.float32)
            
            modelo.eval()
            with torch.no_grad():
                previsoes = (modelo(X_t_test) >= 0.5).float().cpu().numpy()
                f1 = f1_score(y_test, previsoes)
                
            resultados_f1.append({
                "Configuração": nome,
                "Repetição": i + 1,
                "F1-Score": f1
            })
            print(f"Resultado Final F1 (Repetição {i+1}): {f1:.4f}")

    df_resultados = pd.DataFrame(resultados_f1)
    df_resultados.to_csv("resultados_experimentos.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Configuração", y="F1-Score", data=df_resultados)
    plt.title("Comparação de Desempenho (F1-Score) - 10 Execuções por Estratégia")
    plt.ylabel("F1-Score (Maior é melhor)")
    plt.xlabel("Estratégia de Gradiente Descendente")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig("boxplot_comparacao.png")
    print("\n✅ EXPERIMENTOS CONCLUÍDOS!")
    print("✅ Gráfico 'boxplot_comparacao.png' gerado com sucesso!")
    plt.show()

if __name__ == "__main__":
   
    executar_experimentos()