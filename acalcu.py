import cupy as cp
import os

# 1️⃣ 确定当前脚本目录
current_dir = "/mnt/data/jlh/programs/seanet/"
file_dir="SEAnet-main-gpu-copy-8-pretrain/512-ini/"

# 2️⃣ 数据集名称列表
datasets = [
    {"name": "Astro", "compare_file": "astro-compare-10kquery/原版-astro-results.txt"},
    {"name": "Deep1B", "compare_file": "adeep1b-compare/原版-deep1b-results.txt"},
    {"name": "F5", "compare_file": "aF5-compare/原版-F5-results.txt"},
    {"name": "F10", "compare_file": "aF10-compare/原版-F10-results.txt"},
    {"name": "origin", "compare_file": "aorigin-compare/原版-origin-results.txt"},
    {"name": "sald", "compare_file": "asald-compare/原版-sald-results.txt"},
    {"name": "seismic", "compare_file": "aseismic-compare/原版-seismic-results.txt"},
]
# 3️⃣ 其他参数
k = 10  # 最近邻个数
dim = 16

# 4️⃣ 数据读取函数
def load_data(data_file, query_file):
    data_offset = 0
    query_offset = 0
    data = cp.fromfile(data_file, dtype=cp.float32, count=20000 * dim, offset=data_offset).reshape(-1, dim)
    query = cp.fromfile(query_file, dtype=cp.float32, count=1000 * dim, offset=query_offset).reshape(-1, dim)
    return data, query

# 5️⃣ 计算最近邻
def knn_search(data, query, k):
    distances = cp.linalg.norm(data[:, cp.newaxis] - query, axis=2)
    knn_indices = cp.argsort(distances, axis=0)[:k]
    knn_distances = cp.take_along_axis(distances, knn_indices, axis=0)
    return knn_indices, knn_distances

# 6️⃣ 将结果写入文件
def write_results_to_file(knn_indices, knn_distances, k, output_file):
    with open(output_file, 'w') as f:
        for query_idx in range(knn_indices.shape[1]):
            for nn_idx in range(k):
                distance = knn_distances[nn_idx, query_idx]
                index = knn_indices[nn_idx, query_idx]
                f.write(f"the [{query_idx}] query [{nn_idx}] NN is {distance:.6f} at {index}\n")
# 7️⃣ 读取并解析文件
def read_nn_data(file_path):
    nn_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            query_index = int(parts[1].strip('[]'))
            nn_index = int(parts[3].strip('[]'))
            nn_value = float(parts[6])
            nn_position = int(parts[8])
            if query_index not in nn_data:
                nn_data[query_index] = []
            nn_data[query_index].append((nn_index, nn_value, nn_position))
    return nn_data

# 8️⃣ 比较最近邻位置
def compare_nn_positions(nn_data1, nn_data2):
    similarity_ratios = {}
    for query_index in nn_data1:
        if query_index in nn_data2:
            nn_positions1 = [nn[2] for nn in nn_data1[query_index]]
            nn_positions2 = [nn[2] for nn in nn_data2[query_index]]
            common_positions = set(nn_positions1) & set(nn_positions2)
            similarity_ratio = len(common_positions) / len(nn_positions1)
            similarity_ratios[query_index] = similarity_ratio
    return similarity_ratios


# 9️⃣ 主程序
def main(data_file, query_file, k, output_file, compare_file):
    # 计算最近邻
    data, query = load_data(data_file, query_file)
    knn_indices, knn_distances = knn_search(data, query, k)
    write_results_to_file(knn_indices, knn_distances, k, output_file)
    
    # 读取并对比最近邻数据
    nn_data_file1 = read_nn_data(compare_file)
    nn_data_file2 = read_nn_data(output_file)
    similarity_ratios = compare_nn_positions(nn_data_file1, nn_data_file2)
    mean_similarity_ratio = sum(similarity_ratios.values()) / len(similarity_ratios)
    
    return similarity_ratios, mean_similarity_ratio

# 🔟 循环处理每个数据集
for dataset in datasets:
    dataset_name = dataset["name"]
    compare_file = os.path.join(current_dir, f'{dataset["compare_file"]}')
    data_file = os.path.join(current_dir, file_dir,f'{dataset_name}-database.bin')
    query_file = os.path.join(current_dir,file_dir, f'{dataset_name}-query.bin')
    output_file = os.path.join(current_dir,file_dir, f'result/{dataset_name}-results.txt')
    
    # 创建结果文件夹（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Processing dataset: {dataset_name}")
    similarity_ratios, mean_similarity_ratio = main(data_file, query_file, k, output_file, compare_file)
    
    # 输出相似度结果
    with open(output_file, 'a') as f:
        f.write("\nSimilarity Ratios:\n")
        for query_index, ratio in similarity_ratios.items():
            f.write(f"Query {query_index}: Similarity Ratio = {ratio}\n")
        f.write(f"\nMean Similarity Ratio = {mean_similarity_ratio}\n")
    
    print(f"Results written to {output_file}")
    print(f"Mean Similarity Ratio for {dataset_name}: {mean_similarity_ratio:.6f}\n")
    # 定义汇总文件路径
    summary_file = os.path.join(current_dir,file_dir, "1-2-mean_similarity_ratios_summary.txt")

    # 将 Mean Similarity Ratio 写入汇总文件
    with open(summary_file, 'a') as summary:
        summary.write(f"Dataset: {dataset_name}, Mean Similarity Ratio = {mean_similarity_ratio:.6f}\n")

    print(f"Mean Similarity Ratio for {dataset_name} written to summary file.\n")

print("All datasets processed.")