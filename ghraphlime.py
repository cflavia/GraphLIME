import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
from graphlime import GraphLIME

df = pd.read_csv(r"diabetes.csv")
df = df.fillna(df.mean())
X = df.drop(columns=['Outcome'])
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

node_features = torch.tensor(X_scaled, dtype=torch.float)

n_neighbors = 5
knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)

edge_index = []
for i, neighbors in enumerate(indices):
    for neighbor in neighbors:
        if i != neighbor:
            edge_index.append([i, neighbor])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

graph_data = Data(x=node_features, edge_index=edge_index)

class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

input_dim = graph_data.x.shape[1]
hidden_dim = 16
output_dim = 2

model = GCNModel(input_dim, hidden_dim, output_dim)

class SimpleGCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

model = SimpleGCNModel(input_dim, hidden_dim, output_dim)

train_mask, test_mask = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)

labels = torch.tensor(y.values, dtype=torch.long)

train_mask = torch.tensor(train_mask, dtype=torch.long)
test_mask = torch.tensor(test_mask, dtype=torch.long)

graph_data.y = labels
graph_data.train_mask = train_mask
graph_data.test_mask = test_mask

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train(model, graph_data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(graph_data.x, graph_data.edge_index)
    loss = F.nll_loss(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, graph_data):
    model.eval()
    out = model(graph_data.x, graph_data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[graph_data.test_mask] == graph_data.y[graph_data.test_mask]).sum().item()
    accuracy = correct / graph_data.test_mask.sum().item()
    return accuracy

num_epochs = 100
for epoch in range(num_epochs):
    loss = train(model, graph_data, optimizer)

explainer = GraphLIME(model, hop=2, rho=0.1)

node_idx = 0
coefs = explainer.explain_node(node_idx, graph_data.x, graph_data.edge_index)
mean_coefs = coefs.mean()

all_coefs = []
for node_idx in range(graph_data.x.shape[0]):
    coefs = explainer.explain_node(node_idx, graph_data.x, graph_data.edge_index)
    coefs_tensor = torch.tensor(coefs, dtype=torch.float32)
    all_coefs.append(coefs_tensor)

all_coefs = torch.stack(all_coefs)
mean_coefs_per_feature = all_coefs.mean(dim=0)

mean_coefs_dict = {feature_name: mean_coef.item() for feature_name, mean_coef in zip(feature_names, mean_coefs_per_feature)}

threshold  = 0.1

filtered_mean_coefs_dict = {feature: coef for feature, coef in mean_coefs_dict.items() if coef > threshold}

selected_features = list(filtered_mean_coefs_dict.keys())

df = pd.read_csv(r"data.csv")

X = df[selected_features].values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


class AttentionModel(nn.Module):
    def __init__(self, input_dim, attention_dim, hidden_dim, output_dim):
        super(AttentionModel, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim, attention_dim))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        attention_scores = torch.matmul(x, self.attention_weights)
        attention_scores = torch.softmax(attention_scores, dim=1)
        attended_features = x * attention_scores
        x = torch.relu(self.fc1(attended_features))
        x = self.fc2(x)
        return x

model = AttentionModel(input_dim=len(selected_features), attention_dim=1, hidden_dim=64, output_dim=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
