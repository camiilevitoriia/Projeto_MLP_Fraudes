# 🛡️ Detecção de Fraudes Financeiras com Redes Neurais (MLP)

![Status](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)

Projeto prático de Inteligência Computacional desenvolvido na **Universidade Federal do Rio Grande do Norte (UFRN)** para a construção, treinamento e análise de um sistema preditivo antifraude utilizando *Deep Learning*.

## 📋 Resumo Executivo
O projeto implementa uma arquitetura de Perceptron de Múltiplas Camadas (MLP) em PyTorch para classificar transações financeiras bancárias como legítimas ou fraudulentas. O sistema utiliza técnicas de validação cruzada (K-Fold), pipelines rígidos de pré-processamento e estratégias de regularização (Dropout e Early Stopping) para maximizar o F1-Score em um cenário de alta complexidade.

## 👥 Equipe
* **Discentes:** Kayron Nilton da Silva Gomes & Camile Vitoria Gomes da Silva
* **Docente:** Antonino Alves Feitosa Neto
* **Disciplina:** Inteligência Computacional
* **Curso:** Tecnologia em Análise e Desenvolvimento de Sistemas (TADS)


## 🎯 Objetivos
* **Geral:** Construir e avaliar o desempenho de redes neurais artificiais na detecção de anomalias financeiras.
* **Específicos:**
    * Tratar e balancear dados transacionais estruturados.
    * Implementar um MLP modular utilizando a biblioteca PyTorch.
    * Realizar experimentações variando capacidade de neurônios e taxas de Dropout.
    * Extrair, comparar e apresentar métricas focadas em cenários desbalanceados (Recall e F1-Score).

## ⚙️ Arquitetura e Fluxo do Sistema

1.  **Amostragem (`dados.py`):** Coleta orientada por semente fixa (Matrícula) gerando um dataset representativo de 3.000 instâncias.
2.  **Pipeline de Dados (`pre_processamento.py`):**
    * *Numéricos:* Tratamento de nulos via Mediana + StandardScaler.
    * *Categóricos:* Tratamento de nulos via Moda + OneHotEncoding.
3.  **Arquitetura do Modelo (`rede_neural.py`):** Rede Feedforward com ativação ReLU oculta, aplicação opcional de Dropout e camada sigmoidal final.
4.  **Treinamento (`treino.py`):** Treinamento com 5-Fold Stratified Cross-Validation, otimização SGD e salvamento inteligente do melhor estado por *Early Stopping*.
5.  **Análise Fatorial (`main.py`):** Bateria de 30 execuções simulando 3 configurações diferentes.

## 📂 Estrutura de Diretórios
```text
Projeto_MLP_Fraudes/
├── dados.py                 # Funções de sampling e carga de dados
├── pre_processamento.py     # Construção do pipeline sklearn
├── rede_neural.py           # Classe da arquitetura MLP (PyTorch)
├── treino.py                # Lógica de validação cruzada e treinamento
├── main.py                  # Script principal e testes de hipóteses
├── requirements.txt         # Dependências do projeto
├── paysim_sample.csv        # Dataset de amostra gerado localmente
└── resultados/              # (Gerado automaticamente)
    ├── tabela_metricas_completa.csv
    ├── resultados_experimentos.csv
    ├── boxplot_comparacao.png
    └── curvas_aprendizado.png