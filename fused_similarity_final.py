import numpy as np
import pandas as pd

# --- Bước 1: Đọc dữ liệu ---
# Chú ý: Bạn cần có sẵn 6 file tương ứng (có thể từ Excel hoặc .npy)

LFM1 = pd.read_excel('lncRNA_functional_similarity.xlsx', index_col=0).values
LGM1 = pd.read_excel('lncRNA_gaussian_similarity.xlsx', index_col=0).values

MFM1 = pd.read_excel('miRNA_functional_similarity.xlsx', index_col=0).values
MGM1 = pd.read_excel('miRNA_gaussian_similarity.xlsx', index_col=0).values

DSM1 = pd.read_excel('dataset3\interaction\di_di_semantics.xlsx', index_col=0).values
DGM1 = pd.read_excel('disease_gaussian_similarity.xlsx', index_col=0).values

# --- Bước 2: Hàm chuẩn hóa và tạo ma trận tổng hợp ---
def fuse_similarity(A, B):
    C = A + B
    max_val = np.max(C)
    if max_val == 0:
        return np.zeros_like(C)
    return C / max_val

# --- Bước 3: Tạo ma trận LM1, MM1, DM1 ---
LM1 = fuse_similarity(LFM1, LGM1)
MM1 = fuse_similarity(MFM1, MGM1)
DM1 = fuse_similarity(DSM1, DGM1)

# --- Bước 4: Cập nhật (lần thứ nhất) các ma trận đặc trưng ---
LFM2 = LM1 * LFM1
LGM2 = LM1 * LGM1
MFM2 = MM1 * MFM1
MGM2 = MM1 * MGM1
DSM2 = DM1 * DSM1
DGM2 = DM1 * DGM1

# --- Bước 5: Tạo ma trận LM2, MM2, DM2 ---
LM2 = fuse_similarity(LFM2, LGM2)
MM2 = fuse_similarity(MFM2, MGM2)
DM2 = fuse_similarity(DSM2, DGM2)

# --- Bước 6: Cập nhật lần tiếp theo nếu cần (deep extraction tiếp tục như vậy)

# --- Bước 7: Lưu ra file kết quả ---
pd.DataFrame(LM2).to_excel("lncRNA_fused_similarity_final.xlsx", index=False, header=False)
pd.DataFrame(MM2).to_excel("miRNA_fused_similarity_final.xlsx", index=False, header=False)
pd.DataFrame(DM2).to_excel("disease_fused_similarity_final.xlsx", index=False, header=False)

print("✅ Đã hoàn thành trích xuất đặc trưng topo sâu và lưu file!")

