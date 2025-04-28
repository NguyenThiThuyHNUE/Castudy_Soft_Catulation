import obonet
import networkx as nx
import pandas as pd
from tqdm import tqdm

# Tải ontology
url = 'https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo'
G = obonet.read_obo(url)

# Lọc các bệnh liên quan tới ung thư
cancer_nodes = []
for node, data in G.nodes(data=True):
    if node.startswith('DOID'):
        name = data.get('name', '').lower()
        if 'cancer' in name:
            cancer_nodes.append(node)

print(f"✅ Số lượng bệnh ung thư tìm thấy: {len(cancer_nodes)}")

# Tham số attenuation (λ)
lambda_factor = 0.5

# Lấy danh sách DOID (chỉ bệnh ung thư)
disease_nodes = cancer_nodes

# Tiền xử lý: lưu tổ tiên & cây con
ancestors = {d: nx.ancestors(G, d) | {d} for d in disease_nodes}
children = {d: list(G.successors(d)) for d in G.nodes}

# Tính Wdi(d): semantic contribution từ node d tới bệnh di
def compute_W(di):
    memo = {}

    def helper(d):
        if d == di:
            return 1.0
        if d in memo:
            return memo[d]
        child_scores = [lambda_factor * helper(child) for child in children.get(d, []) if child in all_nodes]
        score = max(child_scores, default=0.0)
        memo[d] = score
        return score

    all_nodes = ancestors[di]
    return {d: helper(d) for d in all_nodes}

# Tính toàn bộ W và DS
W_dict = {}
DS_dict = {}

print("Tính Wdi và DS(di)...")
for di in tqdm(disease_nodes):
    Wdi = compute_W(di)
    W_dict[di] = Wdi
    DS_dict[di] = sum(Wdi.values())

# Tính DSim
print("Tính ma trận tương đồng DSim...")
sim_matrix = pd.DataFrame(index=disease_nodes, columns=disease_nodes)

for i, di in tqdm(enumerate(disease_nodes), total=len(disease_nodes)):
    Wdi = W_dict[di]
    DS_i = DS_dict[di]
    for j in range(i, len(disease_nodes)):
        dj = disease_nodes[j]
        Wdj = W_dict[dj]
        DS_j = DS_dict[dj]
        common_ancestors = set(Wdi.keys()) & set(Wdj.keys())
        numerator = sum(Wdi[d] + Wdj[d] for d in common_ancestors)
        denominator = DS_i + DS_j
        sim = numerator / denominator if denominator != 0 else 0.0
        sim_matrix.loc[di, dj] = sim
        sim_matrix.loc[dj, di] = sim  # đối xứng

# Gán đường chéo = 1
for d in disease_nodes:
    sim_matrix.loc[d, d] = 1.0

# Ghi ra Excel
sim_matrix.to_excel("disease_semantic_similarity_cancer_only.xlsx")
print("✅ Đã lưu ma trận chỉ cho bệnh ung thư ra file Excel.")

# import obonet
# import networkx as nx
# import pandas as pd
# from tqdm import tqdm
#
# # Tải ontology
# url = 'https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo'
# G = obonet.read_obo(url)
#
# # Tham số attenuation (λ)
# lambda_factor = 0.5
#
# # Lấy danh sách DOID
# disease_nodes = [n for n in G.nodes if n.startswith('DOID')]
#
# # Tiền xử lý: lưu tổ tiên & cây con
# ancestors = {d: nx.ancestors(G, d) | {d} for d in disease_nodes}
# children = {d: list(G.successors(d)) for d in G.nodes}
#
# # Tính Wdi(d): semantic contribution từ node d tới bệnh di
# def compute_W(di):
#     memo = {}
#
#     def helper(d):
#         if d == di:
#             return 1.0
#         if d in memo:
#             return memo[d]
#         child_scores = [lambda_factor * helper(child) for child in children.get(d, []) if child in all_nodes]
#         score = max(child_scores, default=0.0)
#         memo[d] = score
#         return score
#
#     all_nodes = ancestors[di]
#     return {d: helper(d) for d in all_nodes}
#
# # Tính toàn bộ W và DS
# W_dict = {}
# DS_dict = {}
#
# print("Tính Wdi và DS(di)...")
# for di in tqdm(disease_nodes):
#     Wdi = compute_W(di)
#     W_dict[di] = Wdi
#     DS_dict[di] = sum(Wdi.values())
#
# # Tính DSim
# print("Tính ma trận tương đồng DSim...")
# sim_matrix = pd.DataFrame(index=disease_nodes, columns=disease_nodes)
#
# for i, di in tqdm(enumerate(disease_nodes), total=len(disease_nodes)):
#     Wdi = W_dict[di]
#     DS_i = DS_dict[di]
#     for j in range(i, len(disease_nodes)):
#         dj = disease_nodes[j]
#         Wdj = W_dict[dj]
#         DS_j = DS_dict[dj]
#         common_ancestors = set(Wdi.keys()) & set(Wdj.keys())
#         numerator = sum(Wdi[d] + Wdj[d] for d in common_ancestors)
#         denominator = DS_i + DS_j
#         sim = numerator / denominator if denominator != 0 else 0.0
#         sim_matrix.loc[di, dj] = sim
#         sim_matrix.loc[dj, di] = sim  # đối xứng
#
# # Gán đường chéo = 1
# for d in disease_nodes:
#     sim_matrix.loc[d, d] = 1.0
#
# # Ghi ra Excel
# sim_matrix.to_excel("disease_semantic_similarity_wang.xlsx")
# print("✅ Đã lưu ma trận ra file Excel.")


