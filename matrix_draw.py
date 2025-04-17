import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Số lượng mỗi loại node
n_lnc = 4
n_dis = 4
n_mir = 4

# Tên các node
lncRNAs = [f"lncRNA_{i}" for i in range(n_lnc)]
diseases = [f"disease_{i}" for i in range(n_dis)]
miRNAs = [f"miRNA_{i}" for i in range(n_mir)]

# Ma trận kết hợp
LD = np.array([
    [1, 0.1, 0.3, 0.9],
    [0.1, 0.5, 0.7, 0.2],
    [0.5, 0.6, 0.3, 0.4],
    [0.7, 0.3, 0.9, 1]
])

LM = np.array([
    [0.6, 0.1, 1, 0.9],
    [0, 0.8, 0.7, 0.2],
    [0, 0.6, 0.6, 0.3],
    [0.7, 0.2, 1, 0.7]
])

DM = np.array([
    [0.8, 0.5, 0.3, 0.8],
    [0.4, 0.8, 0, 0.3],
    [0.5, 0, 0.2, 0.4],
    [0.7, 0.7, 1, 0.1]
])

# Ma trận tương đồng nội loại (ngẫu nhiên, có giá trị > 0.3)
LMi = np.random.rand(n_lnc, n_lnc)
DMi = np.random.rand(n_dis, n_dis)
MMi = np.random.rand(n_mir, n_mir)

# Tạo đồ thị
G = nx.Graph()

# Thêm node
G.add_nodes_from(lncRNAs, type='lncRNA')
G.add_nodes_from(diseases, type='disease')
G.add_nodes_from(miRNAs, type='miRNA')

# Thêm liên kết giữa các loại khác nhau
for i in range(n_lnc):
    for j in range(n_dis):
        if LD[i, j] > 0.5:
            G.add_edge(lncRNAs[i], diseases[j], weight=round(LD[i, j], 2), label="LD")
for i in range(n_lnc):
    for j in range(n_mir):
        if LM[i, j] > 0.5:
            G.add_edge(lncRNAs[i], miRNAs[j], weight=round(LM[i, j], 2), label="LM")
for i in range(n_dis):
    for j in range(n_mir):
        if DM[i, j] > 0.5:
            G.add_edge(diseases[i], miRNAs[j], weight=round(DM[i, j], 2), label="DM")

# Thêm cạnh nội loại nếu tương đồng > 0.3
for i in range(n_lnc):
    for j in range(i+1, n_lnc):
        if LMi[i, j] > 0.5:
            G.add_edge(lncRNAs[i], lncRNAs[j], weight=round(LMi[i, j], 2), label="LMi")

for i in range(n_dis):
    for j in range(i+1, n_dis):
        if DMi[i, j] > 0.5:
            G.add_edge(diseases[i], diseases[j], weight=round(DMi[i, j], 2), label="DMi")

for i in range(n_mir):
    for j in range(i+1, n_mir):
        if MMi[i, j] > 0.5:
            G.add_edge(miRNAs[i], miRNAs[j], weight=round(MMi[i, j], 2), label="MMi")

# Vẽ mạng
pos = nx.spring_layout(G, seed=42)
colors = [("skyblue" if G.nodes[n]["type"] == "lncRNA" else "lightcoral" if G.nodes[n]["type"] == "disease" else "lightgreen") for n in G.nodes()]

plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000, font_size=10)
edge_labels = {
    (u, v): f"{d['label']}:{d['weight']}" if 'weight' in d else d['label']
    for u, v, d in G.edges(data=True)
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')
plt.title("Multi-view Heterogeneous Network (lncRNA - disease - miRNA)")
plt.show()


# ---- CHUYỂN ĐỒ THỊ THÀNH MA TRẬN ----

# Danh sách node theo thứ tự cố định
node_list = list(G.nodes)
n_nodes = len(node_list)

# Ma trận kề
A = nx.to_numpy_array(G, nodelist=node_list, weight='weight')  # lấy theo trọng số nếu có

# Ma trận đặc trưng X: one-hot theo loại node
type_map = {'lncRNA': 0, 'disease': 1, 'miRNA': 2}
X = np.zeros((n_nodes, len(type_map)))

for i, node in enumerate(node_list):
    node_type = G.nodes[node]['type']
    X[i, type_map[node_type]] = 1  # one-hot encoding theo loại

# Lưu ma trận A và X vào file .npy
np.save("adjacency_matrix_A.npy", A)
np.save("feature_matrix_X.npy", X)

# In ra kiểm tra
print("Shape of A (adjacency matrix):", A.shape)
print("A matrix:\n", A)

print("\nShape of X (feature matrix):", X.shape)
print("X matrix:\n", X)
