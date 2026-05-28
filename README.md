# 💳 Detecção de Fraudes Financeiras com MLP (Perceptron Multicamadas)

Projeto desenvolvido para a disciplina de Inteligência Computacional do curso de Análise e Desenvolvimento de Sistemas (UFRN). O objetivo deste projeto é aplicar o aprendizado de máquina supervisionado para classificar transações financeiras como legítimas ou fraudulentas utilizando redes neurais PyTorch.

## 📊 1. Dados Utilizados e Metodologia
A base de dados utilizada foi a **PaySim - Mobile Money Transactions Simulation**. 
* **Amostragem Reprodutível:** A base original foi reduzida para **3.000 instâncias** (2.500 normais e 500 fraudulentas) utilizando uma semente aleatória específica (`2025002299`) para garantir a reprodutibilidade do experimento.
* **Isolamento de Teste (Holdout):** 20% da base foi rigorosamente isolada antes de qualquer processamento para servir como validador final (Cofre).
* **Pipeline Antivazamento (Data Leakage):** Foi construído um `ColumnTransformer` (Scikit-Learn) que aplica imputação de valores faltantes (Mediana/Moda) e `StandardScaler` (Z-score) **estritamente dentro das dobras de treino** do K-Fold (Stratified 5-Fold), garantindo que métricas do conjunto de teste não contaminassem o aprendizado.

## 🧠 2. Arquitetura da Rede Neural e Justificativas Técnicas

A estrutura da rede MLP foi projetada seguindo o princípio da parcimônia (Navalha de Ockham), buscando máxima eficiência para dados tabulares sem incorrer em sobreajuste (Overfitting).

* **Número de Camadas Escondidas (1 Camada):** * *Justificativa:* O dataset é relativamente pequeno (3.000 instâncias) e composto por features tabulares simples. O uso de redes profundas (Deep Learning com múltiplas camadas) forçaria a rede a memorizar os dados de treino instantaneamente, gerando overfitting grave. Uma única camada é matematicamente suficiente para mapear as relações lineares e não lineares deste problema.
* **Número de Neurônios na Camada Oculta (10 Neurônios):**
  * *Justificativa:* Atua como um "gargalo" (bottleneck) de informação. Obriga a rede a extrair apenas os padrões mais cruciais que diferenciam uma fraude de uma transação normal, descartando o ruído das variáveis.
* **Taxa de Aprendizado / Learning Rate (0.01):**
  * *Justificativa:* É o "sweet spot" (ponto ideal) para otimizadores iterativos em dados padronizados com Z-Score. Uma taxa maior (ex: 0.1) faria o gradiente oscilar sem encontrar o mínimo global da função de custo, enquanto uma taxa menor (ex: 0.001) tornaria a convergência computacionalmente inviável dentro do limite de épocas estipulado.
* **Número Máximo de Épocas (50) + Early Stopping:**
  * *Justificativa:* Fixar um número estático de épocas é uma falha de design. Estabeleceu-se 50 épocas como teto, associado a um mecanismo de **Early Stopping** (paciência = 5). Assim, a rede cessa o aprendizado dinamicamente no exato milissegundo em que a métrica de *Loss* de validação começa a subir, blindando o modelo contra a memorização.
* **Função de Ativação (ReLU e Sigmoid na Saída):**
  * *Justificativa:* A função **ReLU** foi empregada na camada oculta por mitigar o problema do desvanecimento do gradiente (*Vanishing Gradient*), permitindo cálculos mais ágeis comparados a ativações clássicas (Tanh). Na camada de saída, o uso da **Sigmoid** é mandatório, pois comprime os "logits" brutos para o intervalo `[0, 1]`, transformando a saída do modelo numa probabilidade direta para a classificação binária.

## 🔬 3. Resultados Experimentais: O Efeito do Lote (Batch Size)

O projeto exige uma análise do comportamento médio da rede frente a diferentes estratégias de otimização estocástica. Os resultados obtidos após 40 execuções independentes demonstraram um forte fenômeno de "diluição de sinal" devido ao desbalanceamento das classes.

### Tabela de Desempenho (F1-Score Médio de 10 Execuções)
| Estratégia de Gradiente | Batch Size | Tempo Médio/Fold | F1-Score Médio |
|-------------------------|------------|------------------|----------------|
| **SGD Estocástico Puro**| 1          | ~35.0s           | **~0.86** |
| **Mini-Batch Pequeno** | 32         | ~1.6s            | ~0.65          |
| **Mini-Batch Grande** | 128        | ~1.3s            | ~0.18          |
| **Batch Completo** | 2400       | ~1.4s            | ~0.28          |

### Análise e Conclusão
A configuração vencedora foi o otimizador **Estocástico puro (SGD com Batch = 1)**. 

* **Por que o SGD venceu?** Ao processar e atualizar os pesos uma transação por vez, a rede sente imediatamente o impacto do "erro" gerado ao classificar incorretamente uma fraude. O modelo ajusta a sua direção rapidamente.
* **Por que o Batch falhou?** Estratégias de lote grande falham em bases altamente desbalanceadas porque calculam a média do erro. Como há 2.000 transações normais e apenas 400 fraudes no treino, o alerta de fraude é diluído e "abafado" pelas transações comuns. O modelo converge rapidamente para um mínimo local viciado onde ele "chuta" que quase todas as transações são normais (resultando em um F1-Score péssimo de 0.28).

O tempo de treinamento do SGD é substancialmente maior, evidenciando o *trade-off* clássico da Engenharia de Software: sacrificou-se tempo computacional para obter precisão cirúrgica na captura de anomalias (Fraudes).

## 🚀 4. Como Executar
1. Clone o repositório.
2. Crie e ative um ambiente virtual (`python -m venv venv` ). Ativa No Windows: (`.\venv\Scripts\activate` ).
3. Instale as dependências: `pip install -r requirements.txt`.
4. Garanta que o arquivo `PS_20174392719_1491204439457_log.csv` original esteja dentro da pasta `dados_log.csv/`.
5. Execute o maestro da aplicação: `python main.py`.
