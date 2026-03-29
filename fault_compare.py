import os
import json
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
SCORE_SUM_ROOT = "/home/mcy/flux/exp_results/score_sum"
IMAGE_ROOT = "/home/mcy/flux/exp_results"
DATASET_ROOT = "/home/mcy/flux/benchmarks/T2I-CompBench/examples/dataset"
CATEGORIES = ["color", "complex", "spatial"]
OUTPUT_ROOT = "/home/mcy/flux/exp_results/Failure_Analysis_Final"
TOP_N = 20  # 每个类别筛选前 20 个退步最明显的案例
# ==========================================

def load_full_prompts(category):
    """从 .txt 加载完整 Prompt，跳过非数字开头的行"""
    path = os.path.join(DATASET_ROOT, f"{category}.txt")
    p_map = {}
    if not os.path.exists(path):
        print(f"!!! 警告: 找不到数据集文件 {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            # 修复之前的 ValueError: 只有第一个部分是数字时才解析
            if len(parts) == 2 and parts[0].isdigit():
                p_map[int(parts[0])] = parts[1]
    return p_map

def create_sbs_image(base_p, adap_p, save_p):
    """无缝拼接左右对比图"""
    try:
        img_b = Image.open(base_p).convert("RGB")
        img_a = Image.open(adap_p).convert("RGB")
        w, h = img_b.size
        merged = Image.new('RGB', (w * 2, h))
        merged.paste(img_b, (0, 0))
        merged.paste(img_a, (w, 0))
        merged.save(save_p)
        return True
    except:
        return False

def process_failures(category):
    print(f"\n>>> 正在分析 [{category.upper()}] 类的负面案例...")
    cat_out_dir = os.path.join(OUTPUT_ROOT, category)
    os.makedirs(cat_out_dir, exist_ok=True)
    
    prompt_map = load_full_prompts(category)
    
    # 1. 加载分数
    score_dir = os.path.join(SCORE_SUM_ROOT, category)
    try:
        base_scores = json.load(open(os.path.join(score_dir, "baseline.json")))
        adap_scores = json.load(open(os.path.join(score_dir, "adaptive.json")))
    except:
        print(f"   [跳过] 找不到分数 JSON 文件。")
        return

    # 2. 计算负分差
    failures = []
    common_files = set(base_scores.keys()).intersection(set(adap_scores.keys()))
    for fname in common_files:
        margin = float(adap_scores[fname]) - float(base_scores[fname])
        if margin < 0: # 只记录退步的情况
            img_id = int(fname.split('_')[0])
            failures.append({
                "id": img_id, "filename": fname, "margin": margin,
                "base": base_scores[fname], "adap": adap_scores[fname],
                "prompt": prompt_map.get(img_id, "Unknown (ID not in txt)") if prompt_map else "Unknown"
            })
    
    # 按退步程度排序（Margin 越负越靠前）
    failures.sort(key=lambda x: x['margin'])
    failures = failures[:TOP_N]

    if not failures:
        print(f"   [提示] 该类别下没有发现负面案例。")
        return

    # 3. 输出 txt 报告和拼图
    report_path = os.path.join(cat_out_dir, "failure_prompts.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== {category.upper()} Failure Cases (Adaptive < Baseline) ===\n\n")
        
        img_cat_root = os.path.join(IMAGE_ROOT, f"T2I_CompBench_{category}")
        
        for i, case in enumerate(tqdm(failures)):
            # 拼图文件名：Rank_ID_Margin.png
            save_name = f"Rank{i+1}_ID{case['id']}_Margin{case['margin']:.2f}.png"
            save_p = os.path.join(cat_out_dir, save_name)
            base_p = os.path.join(img_cat_root, "baseline", "samples", case['filename'])
            adap_p = os.path.join(img_cat_root, "adaptive", "samples", case['filename'])
            
            success = create_sbs_image(base_p, adap_p, save_p)
            if success:
                f.write(f"【Rank {i+1}】\n")
                f.write(f"ID: {case['filename']}\n")
                f.write(f"Full Prompt: {case['prompt']}\n")
                f.write(f"Score: {case['base']:.2f} -> {case['adap']:.2f} (Margin: {case['margin']:.2f})\n")
                f.write("-" * 50 + "\n")

    print(f"   [完成] 报告和拼图已生成至: {cat_out_dir}")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for cat in CATEGORIES:
        process_failures(cat)
    print(f"\n🚀 所有负面案例分析已完成！根目录: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()