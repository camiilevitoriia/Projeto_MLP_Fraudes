import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

# Importamos o pipeline e a rede que criámos nos outros ficheiros
from pre_processamento import construir_pipeline
from rede_neural import MLP

def treinar_modelo_kfold(X_train, y_train, batch_size=32, lr=0.01, max_epochs=100, patience=10):
    print(f"\n--- A INICIAR TREINO (Batch Size: {batch_size}, LR: {lr}) ---")
    
    # 1. Configurar K-Fold Estratificado
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Configuração de dispositivo (GPU se houver, senão CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    melhor_modelo_global = None
    melhor_f1_global = 0.0
    melhor_pipeline = None

    # 2. Iniciar o ciclo das 5 fatias (Folds)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\nA treinar o Fold {fold + 1}/5...")
        
        # Separar dados deste fold
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 3. O SEGREDO DO PROFESSOR: O Pipeline é ajustado (fit) APENAS no treino do fold!
        pipeline = construir_pipeline(X_fold_train)
        X_fold_train_prep = pipeline.fit_transform(X_fold_train)
        X_fold_val_prep = pipeline.transform(X_fold_val) # Apenas transform! Sem Data Leakage!
        
        # Converter para Tensores do PyTorch
        X_t_train = torch.tensor(X_fold_train_prep, dtype=torch.float32).to(device)
        y_t_train = torch.tensor(y_fold_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        X_t_val = torch.tensor(X_fold_val_prep, dtype=torch.float32).to(device)
        y_t_val = torch.tensor(y_fold_val.values, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Criar DataLoader (É aqui que aplicamos Batch, Mini-Batch ou SGD)
        dataset_train = TensorDataset(X_t_train, y_t_train)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        
        # 4. Inicializar a Rede, Função de Custo e Otimizador
        dimensao_entrada = X_t_train.shape[1]
        modelo = MLP(input_dim=dimensao_entrada).to(device)
        
        # Binary Cross Entropy (Exigência do enunciado)
        criterio = nn.BCELoss() 
        # Otimizador SGD (Exigência do enunciado)
        otimizador = optim.SGD(modelo.parameters(), lr=lr)
        
        # Variáveis para Early Stopping
        melhor_loss_val = float('inf')
        epocas_sem_melhoria = 0
        
        # 5. O Ciclo de Épocas (O treino real)
        tempo_inicio = time.time()
        for epoch in range(max_epochs):
            modelo.train() # Modo de treino (ativa Dropout)
            
            for batch_X, batch_y in loader_train:
                otimizador.zero_grad()           # Limpa a memória
                previsoes = modelo(batch_X)      # Tenta adivinhar (Forward)
                erro = criterio(previsoes, batch_y) # Calcula o erro (Loss)
                erro.backward()                  # Descobre quem errou (Backpropagation)
                otimizador.step()                # Ajusta os pesos (Gradiente Descendente)
            
            # Avaliação no fim da época (Validação)
            modelo.eval() # Modo de avaliação (desliga Dropout)
            with torch.no_grad():
                val_previsoes = modelo(X_t_val)
                val_loss = criterio(val_previsoes, y_t_val).item()
            
            # EARLY STOPPING: Verifica se melhorou
            if val_loss < melhor_loss_val:
                melhor_loss_val = val_loss
                epocas_sem_melhoria = 0
            else:
                epocas_sem_melhoria += 1
                
            if epocas_sem_melhoria >= patience:
                print(f"Early Stopping ativado na época {epoch+1}!")
                break
                
        tempo_fim = time.time()
        
        # Calcular F1-Score final deste fold
        modelo.eval()
        with torch.no_grad():
            previsoes_finais = modelo(X_t_val)
            previsoes_binarias = (previsoes_finais >= 0.5).float().cpu().numpy()
            f1_fold = f1_score(y_fold_val, previsoes_binarias)
        
        print(f"Fold {fold+1} concluído em {tempo_fim - tempo_inicio:.2f}s | F1-Score: {f1_fold:.4f}")
        
        # Guardar o melhor modelo de todos os folds
        if f1_fold > melhor_f1_global:
            melhor_f1_global = f1_fold
            melhor_modelo_global = modelo
            melhor_pipeline = pipeline
            
    print(f"\n--- TREINO CONCLUÍDO | Melhor F1-Score na Validação: {melhor_f1_global:.4f} ---")
    return melhor_modelo_global, melhor_pipeline

def testar_modelo_final(modelo, pipeline, X_test, y_test):
    # 6. Avaliar o modelo no "Cofre" (Os 20% nunca vistos)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepara os dados de teste (Apenas transform!)
    X_test_prep = pipeline.transform(X_test)
    X_t_test = torch.tensor(X_test_prep, dtype=torch.float32).to(device)
    
    modelo.eval()
    with torch.no_grad():
        previsoes = modelo(X_t_test)
        previsoes_binarias = (previsoes >= 0.5).float().cpu().numpy()
        
    print("\n=================================")
    print("   RESULTADOS DO TESTE FINAL")
    print("=================================")
    print(f"Acurácia : {accuracy_score(y_test, previsoes_binarias):.4f}")
    print(f"Precisão : {precision_score(y_test, previsoes_binarias):.4f}")
    print(f"Recall   : {recall_score(y_test, previsoes_binarias):.4f}")
    print(f"F1-Score : {f1_score(y_test, previsoes_binarias):.4f}")
    print("\nMatriz de Confusão:")
    
    print(confusion_matrix(y_test, previsoes_binarias))