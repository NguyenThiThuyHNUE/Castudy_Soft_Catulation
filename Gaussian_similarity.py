import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Bước 1: Đọc dữ liệu ---

# Ma trận liên kết LncRNA - Disease (LD): dòng là LncRNA, cột là Disease
LD = pd.read_excel('dataset3\interaction\lnc_di.xlsx', index_col=0)

# Ma trận liên kết miRNA - Disease (MD): dòng là miRNA, cột là Disease
MD = pd.read_excel('dataset3\interaction\mi_di.xlsx', index_col=0)

# Ma trận liên kết Disease - LncRNA (transposed LD nếu cần)
# Trong công thức gốc họ lấy LD dùng cho Disease luôn
# Ở đây tôi mặc định LD là đúng theo chiều (lncRNA x disease)

# --- Bước 2: Hàm tính Gaussian similarity ---

def gaussian_similarity(matrix, axis=0):
    if axis == 0:
        profiles = matrix.values  # mỗi dòng là 1 profile
    else:
        profiles = matrix.values.T  # mỗi dòng là 1 profile (nếu theo cột)

    n = profiles.shape[0]

    # Tính gamma
    norms = np.linalg.norm(profiles, axis=1) ** 2
    gamma = 1 / (np.mean(norms) + 1e-8)  # thêm 1e-8 tránh chia 0

    print(f"Gamma tính được: {gamma:.5f}")

    sim_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Tính Gaussian Similarity"):
        for j in range(i, n):
            diff = profiles[i] - profiles[j]
            sim = np.exp(-gamma * np.sum(diff ** 2))
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # đối xứng

    return pd.DataFrame(sim_matrix, index=matrix.index if axis == 0 else matrix.columns,
                        columns=matrix.index if axis == 0 else matrix.columns)

# --- Bước 3: Tính toán các ma trận Gaussian similarity ---

print("✅ Tính Gaussian Similarity cho LncRNA...")
LGSim = gaussian_similarity(LD, axis=0)

print("✅ Tính Gaussian Similarity cho miRNA...")
MGSim = gaussian_similarity(MD, axis=0)

print("✅ Tính Gaussian Similarity cho Disease...")
DGSim = gaussian_similarity(LD, axis=1)  # Disease theo cột nên axis=1

# --- Bước 4: Lưu ra file Excel ---

LGSim.to_excel("lncRNA_gaussian_similarity.xlsx")
MGSim.to_excel("miRNA_gaussian_similarity.xlsx")
DGSim.to_excel("disease_gaussian_similarity.xlsx")

print("🎯 Đã lưu 3 ma trận Gaussian similarity thành công!")
