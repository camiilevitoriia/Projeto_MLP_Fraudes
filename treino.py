import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader

from pre_processamento import construir_pipeline
from rede_neural import MLP

def treinar_modelo_kfold(X_train, y_train, batch_size=32, lr=0.01, max_epochs=100, patience=10, num_neuronios=10, taxa_dropout=0.3):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    melhor_modelo_global = None
    melhor_f1_global = 0.0
    melhor_pipeline = None
    melhor_historico = None # Para guardar as curvas de evolução

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pipeline = construir_pipeline(X_fold_train)
        X_fold_train_prep = pipeline.fit_transform(X_fold_train)
        X_fold_val_prep = pipeline.transform(X_fold_val)
        
        X_t_train = torch.tensor(X_fold_train_prep, dtype=torch.float32).to(device)
        y_t_train = torch.tensor(y_fold_train.values, dtype=torch.float32).unsqueeze(1).to(device)
        X_t_val = torch.tensor(X_fold_val_prep, dtype=torch.float32).to(device)
        y_t_val = torch.tensor(y_fold_val.values, dtype=torch.float32).unsqueeze(1).to(device)
        
        dataset_train = TensorDataset(X_t_train, y_t_train)
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        
        dimensao_entrada = X_t_train.shape[1]
        modelo = MLP(input_dim=dimensao_entrada, num_neuronios=num_neuronios, taxa_dropout=taxa_dropout).to(device)
        criterio = nn.BCELoss() 
        otimizador = optim.SGD(modelo.parameters(), lr=lr)
        
        melhor_loss_val = float('inf')
        epocas_sem_melhoria = 0
        
        # Histórico desta dobra (Fold)
        historico = {'acc_train': [], 'f1_train': [], 'acc_val': [], 'f1_val': []}
        
        for epoch in range(max_epochs):
            modelo.train()
            train_preds, train_trues = [], []
            
            for batch_X, batch_y in loader_train:
                otimizador.zero_grad()
                previsoes = modelo(batch_X)
                erro = criterio(previsoes, batch_y)
                erro.backward()
                otimizador.step()
                
                train_preds.extend((previsoes >= 0.5).float().cpu().numpy())
                train_trues.extend(batch_y.cpu().numpy())
            
            acc_train = accuracy_score(train_trues, train_preds)
            f1_train = f1_score(train_trues, train_preds, zero_division=0)
            
            modelo.eval()
            with torch.no_grad():
                val_previsoes = modelo(X_t_val)
                val_loss = criterio(val_previsoes, y_t_val).item()
                val_preds_bin = (val_previsoes >= 0.5).float().cpu().numpy()
                
                acc_val = accuracy_score(y_fold_val, val_preds_bin)
                f1_val = f1_score(y_fold_val, val_preds_bin, zero_division=0)
            
            # Guardar os dados para o gráfico do professor
            historico['acc_train'].append(acc_train)
            historico['f1_train'].append(f1_train)
            historico['acc_val'].append(acc_val)
            historico['f1_val'].append(f1_val)
            
            if val_loss < melhor_loss_val:
                melhor_loss_val = val_loss
                epocas_sem_melhoria = 0
            else:
                epocas_sem_melhoria += 1
                
            if epocas_sem_melhoria >= patience:
                break
                
        # Guardar o modelo e o histórico se for o melhor até agora
        if f1_val > melhor_f1_global:
            melhor_f1_global = f1_val
            melhor_modelo_global = modelo
            melhor_pipeline = pipeline
            melhor_historico = historico
            
    # GERAÇÃO DO GRÁFICO EXIGIDO PELO PROFESSOR (Apenas para o melhor fold)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(melhor_historico['acc_train'], label='Treino (Acurácia)')
    plt.plot(melhor_historico['acc_val'], label='Validação (Acurácia)')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Épocas')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(melhor_historico['f1_train'], label='Treino (F1)')
    plt.plot(melhor_historico['f1_val'], label='Validação (F1)')
    plt.title('Evolução do F1-Score')
    plt.xlabel('Épocas')
    plt.legend()
    plt.tight_layout()
    plt.savefig('curvas_aprendizado.png')
    plt.close()
    
    return melhor_modelo_global, melhor_pipeline