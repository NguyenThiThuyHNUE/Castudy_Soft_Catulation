import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# --- Bước 1: Chuyển A và X thành tensor ---
A_np = np.load("adjacency_matrix_A.npy")
X_np = np.load("feature_matrix_X.npy")

A_tensor = torch.tensor(A_np, dtype=torch.float32)
X_tensor = torch.tensor(X_np, dtype=torch.float32)

# --- Bước 2: Tính Laplacian chuẩn hóa ---
def normalize_adjacency(A):
    D = torch.diag(torch.pow(A.sum(1), -0.5))
    D[torch.isinf(D)] = 0
    L = D @ A @ D
    return L

L_norm = normalize_adjacency(A_tensor)

# --- Bước 3: Định nghĩa mô hình GCN-AE ---
class GCN_AE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCN_AE, self).__init__()
        # Encoder
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)
        # Decoder (bilinear-style)
        self.W3 = nn.Linear(out_dim, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, in_dim)

    def forward(self, A_norm, X):
        # Encoder
        h1 = F.relu(self.W1(torch.matmul(A_norm, X)))
        z = self.W2(h1)  # Embedding

        # Decoder
        h2 = F.relu(self.W3(z))
        X_hat = self.W4(h2)  # Reconstruction

        return z, X_hat


# --- Bước 4: Khởi tạo mô hình ---
model = GCN_AE(in_dim=X_tensor.shape[1], hidden_dim=16, out_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# --- Bước 5: Huấn luyện ---
epochs = 200
for epoch in range(epochs):
    model.train()
    z, X_hat = model(L_norm, X_tensor)

    loss = criterion(X_hat, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# --- Đầu ra: đặc trưng nhúng Z ---
print("\nLow-dimensional embeddings (Z):")
print(z.detach().numpy())


# --- Chuyển về NumPy ---
X_input = X_tensor.detach().numpy()
X_output = X_hat.detach().numpy()

# --- Vẽ heatmap cho X và X_hat ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(X_input, cmap='viridis', aspect='auto')
plt.title("Ma trận đầu vào (X)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(X_output, cmap='viridis', aspect='auto')
plt.title("Ma trận tái tạo (X_hat)")
plt.colorbar()

plt.tight_layout()
plt.show()

# --- Lưu X_hat ra file .npy ---
np.save("output_matrix_X_hat.npy", X_output)