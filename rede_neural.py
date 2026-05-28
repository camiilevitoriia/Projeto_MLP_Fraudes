import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, num_neuronios=10):
        super(MLP, self).__init__()
        
        self.camada_oculta = nn.Linear(input_dim, num_neuronios)
        
        self.ativacao = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.3)
        
        self.camada_saida = nn.Linear(num_neuronios, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.camada_oculta(x)
        x = self.ativacao(x)
        x = self.dropout(x)
        x = self.camada_saida(x)
        x = self.sigmoid(x)
        return x