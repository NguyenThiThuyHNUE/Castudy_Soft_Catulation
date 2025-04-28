# Version 3
import numpy as np
import pandas as pd
import os

# --- Bước 1: Đọc file ---
def read_matrix(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")
    return pd.read_excel(filepath, index_col=0).values

# --- Bước 2: Hàm chỉnh kích thước ---
def adjust_matrix(mat, target_rows, name, target_cols=None):
    if mat.shape[0] > target_rows:
        print(f"⚠️ {name} có {mat.shape[0]} dòng, cắt xuống còn {target_rows}.")
        mat = mat[:target_rows, :]
    elif mat.shape[0] < target_rows:
        raise ValueError(f"❌ {name} thiếu dòng!")

    if target_cols is not None:
        if mat.shape[1] > target_cols:
            print(f"⚠️ {name} có {mat.shape[1]} cột, cắt xuống còn {target_cols}.")
            mat = mat[:, :target_cols]
        elif mat.shape[1] < target_cols:
            raise ValueError(f"❌ {name} thiếu cột!")

    return mat

# --- Bước 3: Đường dẫn file ---
folder = r'C:\Users\Teacher\Documents\De_cuong\LDAGMbyTHUY'
interaction_folder = os.path.join(folder, 'dataset3', 'interaction')

L_L_path = os.path.join(folder, 'lncRNA_fused_similarity_final.xlsx')
M_M_path = os.path.join(folder, 'miRNA_fused_similarity_final.xlsx')
D_D_path = os.path.join(folder, 'disease_fused_similarity_final.xlsx')

LM_path = os.path.join(interaction_folder, 'lnc_mi.xlsx')
LD_path = os.path.join(interaction_folder, 'lnc_di.xlsx')
DM_path = os.path.join(interaction_folder, 'mi_di.xlsx')

# --- Bước 4: Đọc các ma trận ---
L_L = read_matrix(L_L_path)
M_M = read_matrix(M_M_path)
D_D = read_matrix(D_D_path)

LM = read_matrix(LM_path)
LD = read_matrix(LD_path)
DM = read_matrix(DM_path)

# --- Bước 5: Xác định số lượng node ---
n_lnc = L_L.shape[0]
n_mir = M_M.shape[0]
n_dis = D_D.shape[0]

print(f"Số lượng lncRNA: {n_lnc}, miRNA: {n_mir}, disease: {n_dis}")

# --- Bước 6: Đồng bộ ma trận tương tác ---
LM = adjust_matrix(LM, target_rows=n_lnc, name="LM (lnc-miRNA)", target_cols=n_mir)
LD = adjust_matrix(LD, target_rows=n_lnc, name="LD (lnc-disease)", target_cols=n_dis)
DM = adjust_matrix(DM, target_rows=n_mir, name="DM (miRNA-disease)", target_cols=n_dis)

# --- Bước 7: Ghép ma trận dị thể ---
top_row = np.hstack([L_L, LM, LD])
mid_row = np.hstack([LM.T, M_M, DM])
bot_row = np.hstack([LD.T, DM.T, D_D])

A = np.vstack([top_row, mid_row, bot_row])

print(f"✅ Kích thước ma trận dị thể A: {A.shape}")

# --- Bước 8: Lưu file nếu cần ---
np.save(os.path.join(folder, 'heterogeneous_network_A.npy'), A)
print("✅ Đã lưu ma trận A vào heterogeneous_network_A.npy")

# --- Nếu muốn lưu thêm file Excel để kiểm tra trực quan ---
# pd.DataFrame(A).to_excel(os.path.join(folder, 'heterogeneous_network_A.xlsx'))
# print("✅ Đã lưu ma trận A vào heterogeneous_network_A.xlsx")



# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# # Đọc ma trận lncRNA-lncRNA
# L_L = pd.read_excel(r'C:\Users\Teacher\Documents\De_cuong\LDAGMbyTHUY\lncRNA_fused_similarity_final.xlsx', index_col=0).values
#
# # Đọc ma trận miRNA-miRNA
# M_M = pd.read_excel(r'C:\Users\Teacher\Documents\De_cuong\LDAGMbyTHUY\miRNA_fused_similarity_final.xlsx', index_col=0).values
#
# # Đọc ma trận disease-disease
# D_D = pd.read_excel(r'C:\Users\Teacher\Documents\De_cuong\LDAGMbyTHUY\disease_fused_similarity_final.xlsx', index_col=0).values
#
# # Số lượng mỗi loại node
# n_lnc = L_L.shape[0]   # hoặc L_L.shape[1], vì là ma trận vuông
# n_mir = M_M.shape[0]
# n_dis = D_D.shape[0]
#
# print(n_lnc,n_mir,n_dis)
#
# # Ma trận kết hợp
# LD = pd.read_excel(r'C:\Users\Teacher\Documents\De_cuong\LDAGMbyTHUY\dataset3\interaction\lnc_di.xlsx', index_col=0).values
#
# LM = pd.read_excel(r'C:\Users\Teacher\Documents\De_cuong\LDAGMbyTHUY\dataset3\interaction\lnc_mi.xlsx', index_col=0).values
#
# DM = pd.read_excel(r'C:\Users\Teacher\Documents\De_cuong\LDAGMbyTHUY\dataset3\interaction\mi_di.xlsx', index_col=0).values
#
#
# # Ma trận dị thể (block matrix)
# top_row = np.hstack([L_L,       LM,        LD])
# mid_row = np.hstack([LM.T,      M_M,        DM])
# bot_row = np.hstack([LD.T,      DM.T,       D_D])
#
# # Kết hợp thành ma trận tổng
# A = np.vstack([top_row, mid_row, bot_row])
#
# # In kích thước
# print("Kích thước ma trận dị thể A:", A.shape)
#
# # Tên các node
# lncRNAs = [f"lncRNA_{i}" for i in range(n_lnc)]
# diseases = [f"disease_{i}" for i in range(n_dis)]
# miRNAs = [f"miRNA_{i}" for i in range(n_mir)]

# # Ma trận tương đồng nội loại (ngẫu nhiên, có giá trị > 0.3)
# LMi = np.random.rand(n_lnc, n_lnc)
# DMi = np.random.rand(n_dis, n_dis)
# MMi = np.random.rand(n_mir, n_mir)

# Tạo đồ thị
# G = nx.Graph()
#
# # Thêm node
# G.add_nodes_from(lncRNAs, type='lncRNA')
# G.add_nodes_from(diseases, type='disease')
# G.add_nodes_from(miRNAs, type='miRNA')
#
# # Thêm liên kết giữa các loại khác nhau
# for i in range(n_lnc):
#     for j in range(n_dis):
#         if LD[i, j] > 0.5:
#             G.add_edge(lncRNAs[i], diseases[j], weight=round(LD[i, j], 2), label="LD")
# for i in range(n_lnc):
#     for j in range(n_mir):
#         if LM[i, j] > 0.5:
#             G.add_edge(lncRNAs[i], miRNAs[j], weight=round(LM[i, j], 2), label="LM")
# for i in range(n_dis):
#     for j in range(n_mir):
#         if DM[i, j] > 0.5:
#             G.add_edge(diseases[i], miRNAs[j], weight=round(DM[i, j], 2), label="DM")
#
# # Thêm cạnh nội loại nếu tương đồng > 0.3
# for i in range(n_lnc):
#     for j in range(i+1, n_lnc):
#         if LMi[i, j] > 0.5:
#             G.add_edge(lncRNAs[i], lncRNAs[j], weight=round(LMi[i, j], 2), label="LMi")
#
# for i in range(n_dis):
#     for j in range(i+1, n_dis):
#         if DMi[i, j] > 0.5:
#             G.add_edge(diseases[i], diseases[j], weight=round(DMi[i, j], 2), label="DMi")
#
# for i in range(n_mir):
#     for j in range(i+1, n_mir):
#         if MMi[i, j] > 0.5:
#             G.add_edge(miRNAs[i], miRNAs[j], weight=round(MMi[i, j], 2), label="MMi")
#
# # Vẽ mạng
# pos = nx.spring_layout(G, seed=42)
# colors = [("skyblue" if G.nodes[n]["type"] == "lncRNA" else "lightcoral" if G.nodes[n]["type"] == "disease" else "lightgreen") for n in G.nodes()]
#
# plt.figure(figsize=(12, 10))
# nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1000, font_size=10)
# edge_labels = {
#     (u, v): f"{d['label']}:{d['weight']}" if 'weight' in d else d['label']
#     for u, v, d in G.edges(data=True)
# }
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')
# plt.title("Multi-view Heterogeneous Network (lncRNA - disease - miRNA)")
# plt.show()
#
#
# # ---- CHUYỂN ĐỒ THỊ THÀNH MA TRẬN ----
#
# # Danh sách node theo thứ tự cố định
# node_list = list(G.nodes)
# n_nodes = len(node_list)
#
# # Ma trận kề
# A = nx.to_numpy_array(G, nodelist=node_list, weight='weight')  # lấy theo trọng số nếu có
#
# # Ma trận đặc trưng X: one-hot theo loại node
# type_map = {'lncRNA': 0, 'disease': 1, 'miRNA': 2}
# X = np.zeros((n_nodes, len(type_map)))
#
# for i, node in enumerate(node_list):
#     node_type = G.nodes[node]['type']
#     X[i, type_map[node_type]] = 1  # one-hot encoding theo loại
#
# # Lưu ma trận A và X vào file .npy
# np.save("adjacency_matrix_A.npy", A)
# np.save("feature_matrix_X.npy", X)
#
# # In ra kiểm tra
# print("Shape of A (adjacency matrix):", A.shape)
# print("A matrix:\n", A)
#
# print("\nShape of X (feature matrix):", X.shape)
# print("X matrix:\n", X)
