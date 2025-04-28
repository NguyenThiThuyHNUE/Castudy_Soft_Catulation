import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Bước 1: Đọc dữ liệu ---

# Ma trận độ tương đồng ngữ nghĩa bệnh (Disease-Disease Semantic Similarity)
DSim = pd.read_excel('dataset3\interaction\di_di_semantics.xlsx', index_col=0)

# Ma trận liên kết LncRNA-Bệnh
Lnc_Di = pd.read_excel('dataset3\interaction\lnc_di.xlsx', index_col=0)

# Ma trận liên kết miRNA-Bệnh
Mi_Di = pd.read_excel('dataset3\interaction\mi_di.xlsx', index_col=0)


# --- Bước 2: Hàm tính tương đồng chức năng ---

def calculate_functional_similarity(assoc_matrix, DSim_matrix):
    nodes = assoc_matrix.index.tolist()
    diseases = assoc_matrix.columns.tolist()

    sim_matrix = pd.DataFrame(index=nodes, columns=nodes, dtype=float)

    for i in tqdm(range(len(nodes)), desc="Tính tương đồng"):
        for j in range(i, len(nodes)):
            Ri = assoc_matrix.iloc[i]
            Rj = assoc_matrix.iloc[j]

            # Các bệnh liên kết với Ri và Rj
            d1_list = [d for d in diseases if Ri[d] > 0]
            d2_list = [d for d in diseases if Rj[d] > 0]

            if len(d1_list) == 0 or len(d2_list) == 0:
                sim = 0.0
            else:
                # Phía Ri: với mỗi bệnh d1i, lấy max(Dsim(d1i, d2j))
                sum1 = sum([max([DSim_matrix.loc[d1i, d2j] for d2j in d2_list]) for d1i in d1_list])
                # Phía Rj: với mỗi bệnh d2j, lấy max(Dsim(d2j, d1i))
                sum2 = sum([max([DSim_matrix.loc[d2j, d1i] for d1i in d1_list]) for d2j in d2_list])

                sim = (sum1 + sum2) / (len(d1_list) + len(d2_list))

            sim_matrix.iloc[i, j] = sim
            sim_matrix.iloc[j, i] = sim  # Ma trận đối xứng

    return sim_matrix


# --- Bước 3: Tính toán ---

print("Đang tính toán ma trận tương đồng chức năng LncRNA-LncRNA...")
FSim_Lnc = calculate_functional_similarity(Lnc_Di, DSim)

print("Đang tính toán ma trận tương đồng chức năng miRNA-miRNA...")
FSim_Mi = calculate_functional_similarity(Mi_Di, DSim)

# --- Bước 4: Lưu ra file Excel ---

FSim_Lnc.to_excel("lncRNA_functional_similarity.xlsx")
FSim_Mi.to_excel("miRNA_functional_similarity.xlsx")

print("✅ Đã lưu 2 ma trận kết quả ra file Excel.")
