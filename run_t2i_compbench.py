import os
from tqdm import tqdm
# 假设你的推理函数在这个文件里
from flux_pt_acfg import run_inference 

def main():
    # 1. 路径配置
    prompt_dir = "/home/mcy/flux/benchmarks/T2I-CompBench/examples/dataset"
    subset = "spatial"
    prompt_file = os.path.join(prompt_dir, f"{subset}.txt")
    
    # 2. 结果根目录
    output_root = f"/home/mcy/flux/exp_results/T2I_CompBench_{subset}"
    
    # 3. 读取 Prompt
    with open(prompt_file, 'r') as f:
        # 官方文件通常一行一个 Prompt
        prompts = [line.strip() for line in f.readlines() if line.strip()]

    # 4. 设定跑 600 个
    num_samples = 200
    if len(prompts) < num_samples:
        num_samples = len(prompts)
        print(f"警告：Prompt 文件只有 {len(prompts)} 行，将全部跑完。")

    # 5. 循环两种方法
    for method in ["baseline", "adaptive"]:
        # 核心修改：直接在路径里加上 /samples/，对齐评测脚本的要求
        method_samples_dir = os.path.join(output_root, method, "samples")
        os.makedirs(method_samples_dir, exist_ok=True)
        
        print(f"\n>>> 正在使用 [{method}] 方案生成 {num_samples} 张图片...")
        print(f">>> 存储路径: {method_samples_dir}")

        for i in tqdm(range(num_samples)):
            p = prompts[i]
            
            # 文件名处理：001_A_red_ball.png
            # 加上序号 i 是为了方便评测脚本排序
            clean_p = p.replace(" ", "_").replace(".", "").replace("/", "")[:50]
            save_path = os.path.join(method_samples_dir, f"{i:03d}_{clean_p}.png")
            
            # 如果文件已存在则跳过（方便断点续传）
            if os.path.exists(save_path):
                continue
                
            run_inference(p, method=method, save_path=save_path)

    print("\n[完成] 600 个样本已全部生成，你可以直接开始运行 BLIP_vqa.py 了！")

if __name__ == "__main__":
    main()