import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from graphlime import GraphLIME
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

data = pd.read_csv('data.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

k = 5
adjacency_matrix = kneighbors_graph(X_train, n_neighbors=k, mode='connectivity', include_self=True)
edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)

x = torch.tensor(X_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.float32)

data_graph = Data(x=x, edge_index=edge_index, y=y)

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

input_dim = X_train.shape[1]
hidden_dim = 32
output_dim = X_train.shape[1]

model = GCN(in_channels=input_dim, hidden_channels=hidden_dim, out_channels=output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data_graph.x, data_graph.edge_index).squeeze()
    loss = criterion(out, data_graph.x)
    loss.backward()
    optimizer.step()

class CustomGraphLIME(GraphLIME):
    def __init__(self, model, hop=2, rho=0.1):
        super().__init__(model, hop, rho)

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        from sklearn.linear_model import LassoLars

        n, d = x.size()
        dist = (x.unsqueeze(0) - x.unsqueeze(1)).pow(2).sum(dim=-1).sqrt()
        K = torch.exp(-dist / (2 * self.rho ** 2))

        K_bar = K[node_idx].detach().numpy()[:d].reshape(-1, 1)
        L_bar = x[node_idx].detach().numpy()

        if K_bar.shape[0] != len(L_bar):
            K_bar = K_bar[:len(L_bar)]

        solver = LassoLars(alpha=self.rho, fit_intercept=False, positive=True)
        solver.fit(K_bar, L_bar)
        coefs = solver.coef_

        if len(coefs) < d:
            coefs = np.pad(coefs, (0, d - len(coefs)), 'constant')

        return torch.tensor(coefs, dtype=torch.float32)

explainer = CustomGraphLIME(model=model, hop=2, rho=0.1)

node_idx = 0
coefs = explainer.explain_node(node_idx, data_graph.x, data_graph.edge_index)

contributions_df = pd.DataFrame({
    'Feature': data.columns[:-1],
    'Contribution': coefs.detach().numpy()
})

contributions_df = contributions_df.sort_values(by='Contribution', ascending=False)

relevant_features = contributions_df[contributions_df['Contribution'] >= 0]['Feature'].tolist()

relevant_feature_indices = [list(data.columns[:-1]).index(f) for f in relevant_features]

X_train_relevant = X_train[:, relevant_feature_indices]
X_test_relevant = X_test[:, relevant_feature_indices]

X_train_relevant_tensor = torch.tensor(X_train_relevant, dtype=torch.float32)
X_test_relevant_tensor = torch.tensor(X_test_relevant, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class AttentionMechanism(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ComplexAttentionMechanism, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        attention_scores = self.attention_network(x)
        return x * attention_scores

hidden_dim = 16
attention_layer = AttentionMechanism(len(relevant_features), hidden_dim)

attention_output_train = attention_layer(X_train_relevant_tensor)
attention_output_test = attention_layer(X_test_relevant_tensor)

linear_model = nn.Linear(len(relevant_features), 1)
optimizer = optim.Adam(list(linear_model.parameters()) + list(attention_layer.parameters()), lr=0.01)
criterion = nn.MSELoss()

epochs = 100
for epoch in range(epochs):
    linear_model.train()
    attention_layer.train()
    optimizer.zero_grad()

    attention_output_train = attention_layer(X_train_relevant_tensor)

    predictions_train = linear_model(attention_output_train).squeeze()
    loss = criterion(predictions_train, y_train_tensor)

    loss.backward(retain_graph=True)
    optimizer.step()

linear_model.eval()
with torch.no_grad():
    predictions_test = linear_model(attention_output_test).squeeze().detach().numpy()
