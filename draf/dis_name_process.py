import obonet
import pandas as pd
from tqdm import tqdm
from rapidfuzz import process, fuzz

# Tải ontology từ GitHub
url = 'https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo'
G = obonet.read_obo(url)

# Tạo ánh xạ tên bệnh -> DOID
name_to_doid = {}
for node_id, data in G.nodes(data=True):
    if 'name' in data:
        name_to_doid[data['name'].lower()] = node_id

# Chuẩn bị list tên bệnh trong ontology để fuzzy match
ontology_names = list(name_to_doid.keys())

# Đọc file danh sách bệnh (ví dụ cột 'Disease')
input_file = 'dataset1\data_preprocessing\Dis-name-process.xlsx'
df = pd.read_excel(input_file)

# Chuẩn hóa tên bệnh
disease_names = df['Disease'].str.strip().str.lower()


# Hàm tìm DOID (ưu tiên exact match, sau đó fuzzy match)
def find_best_doid(disease_name, threshold=85):
    # Exact match trước
    if disease_name in name_to_doid:
        return name_to_doid[disease_name], "Exact"

    # Fuzzy match nếu không exact
    best_match, score, _ = process.extractOne(
        disease_name, ontology_names, scorer=fuzz.token_sort_ratio
    )
    if score >= threshold:
        return name_to_doid[best_match], f"Fuzzy ({score}%)"
    else:
        return None, "No match"


# Tìm DOID cho từng bệnh
doid_list = []
match_type_list = []
for name in tqdm(disease_names):
    doid, match_type = find_best_doid(name)
    doid_list.append(doid)
    match_type_list.append(match_type)

# Ghi kết quả ra file
df['DOID'] = doid_list
df['MatchType'] = match_type_list
df.to_excel('disease_with_doid_fuzzy.xlsx', index=False)
print("✅ Đã lưu file disease_with_doid_fuzzy.xlsx chứa mã DOID tương ứng và loại khớp.")

# import obonet
# import pandas as pd
# from tqdm import tqdm
#
# # Tải ontology từ GitHub
# url = 'https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo'
# G = obonet.read_obo(url)
#
# # Tạo ánh xạ tên bệnh -> DOID
# name_to_doid = {}
# for node_id, data in G.nodes(data=True):
#     if 'name' in data:
#         name_to_doid[data['name'].lower()] = node_id
#
# # Giả sử bạn có file danh sách bệnh (ví dụ: diseases.xlsx, cột 'Disease')
# input_file = 'dataset1\data_preprocessing\Dis-name-process.xlsx'
# df = pd.read_excel(input_file)
#
# # Chuẩn hóa tên bệnh để tìm
# disease_names = df['Disease'].str.strip().str.lower()
#
# # Tìm DOID tương ứng
# doids = []
# for name in tqdm(disease_names):
#     doid = name_to_doid.get(name, None)
#     doids.append(doid)
#
# # Ghi kết quả ra Excel
# df['DOID'] = doids
# df.to_excel('disease_with_doid.xlsx', index=False)
# print("✅ Đã lưu file disease_with_doid.xlsx chứa mã DOID tương ứng.")
