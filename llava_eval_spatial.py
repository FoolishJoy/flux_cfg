import torch
import os
import json
import re
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig 

# --- 配置路径 ---
BASE_DIR = "/home/mcy/flux/exp_results/T2I_CompBench_spatial/baseline/samples"
ADAP_DIR = "/home/mcy/flux/exp_results/T2I_CompBench_spatial/adaptive/samples"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OUTPUT_JSON = "spatial_eval_paper_summary.json" # 修正了你原脚本保存为 color 的 bug

print(">>> 正在加载 LLaVA-1.5-7B 模型 (采用论文标准 CoT 评测)...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    quantization_config=quantization_config,
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    device_map="auto"
)

def get_score_from_llava(image_path, prompt_text):
    # 【核心升级】：符合论文规范的 CoT (思维链) 与离散量表 (Rubric)
    instruction = (
        f"USER: <image>\nYou are an expert evaluator for text-to-image models. "
        f"Evaluate the spatial relationship based on this prompt: '{prompt_text}'.\n\n"
        "Rubric:\n"
        "3 - Perfect: All objects are present and their spatial relationship exactly matches the prompt.\n"
        "2 - Ambiguous: Objects are present, but the spatial relationship is slightly off or viewed from a confusing angle.\n"
        "1 - Wrong: Objects are present, but the spatial relationship is completely wrong (e.g., left instead of right).\n"
        "0 - Missing: One or more key objects mentioned in the prompt are completely missing.\n\n"
        "Step 1: Briefly describe the position of the objects in the image.\n"
        "Step 2: Provide a final score (0, 1, 2, or 3) based on the rubric.\n"
        "Format your response exactly as follows:\n"
        "Reasoning: [your analysis]\n"
        "Score: [your score]\nASSISTANT:"
    )
    
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(text=instruction, images=raw_image, return_tensors="pt").to("cuda")
        
        # 放宽 max_new_tokens，给它空间输出推理过程
        output = model.generate(**inputs, max_new_tokens=150, temperature=0.2, do_sample=True)
        res_text = processor.decode(output[0], skip_special_tokens=True)
        ans = res_text.split("ASSISTANT:")[-1].strip()
        
        # 提取 'Score: X' 里的数字
        match = re.search(r"Score:\s*([0-3])", ans, re.IGNORECASE)
        if match:
            raw_score = int(match.group(1))
        else:
            # 备用正则：如果在段落末尾只输出了数字
            numbers = re.findall(r"\b[0-3]\b", ans)
            raw_score = int(numbers[-1]) if numbers else 1 # 默认给 1 分(存在但关系错)
            
        # 将 0-3 映射回 0.0 - 1.0 的学术常用区间
        normalized_score = raw_score / 3.0
        return normalized_score, ans
        
    except Exception as e:
        print(f"Error evaluating {image_path}: {e}")
        return 0.33, "Error during inference" # 发生错误给个默认低分

def main():
    print(f">>> 检查路径: {BASE_DIR}")
    base_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith('.png')])
    
    if len(base_files) == 0:
        print("!!! 致命错误：Baseline 文件夹中没有找到 .png 图片，请检查路径。")
        return

    results = []
    base_total = 0
    adap_total = 0
    
    print(f">>> 开始对 {len(base_files)} 个样本进行深度交叉打分...")
    for filename in tqdm(base_files):
        img_id = filename.split('_')[0]
        prompt = " ".join(filename.split('_')[1:]).rsplit('.', 1)[0].replace('_', ' ')
        
        path_b = os.path.join(BASE_DIR, filename)
        path_a = os.path.join(ADAP_DIR, filename)
        
        if not os.path.exists(path_a): 
            print(f"Warning: {filename} 不在 Adaptive 文件夹中，跳过。")
            continue
        
        # 获取得分和推理过程
        score_b, reason_b = get_score_from_llava(path_b, prompt)
        score_a, reason_a = get_score_from_llava(path_a, prompt)
        
        base_total += score_b
        adap_total += score_a
        
        results.append({
            "id": img_id,
            "prompt": prompt,
            "baseline": {
                "score": score_b,
                "reasoning": reason_b
            },
            "adaptive": {
                "score": score_a,
                "reasoning": reason_a
            },
            "margin": score_a - score_b
        })

    # 计算最终统计
    n = len(results)
    if n == 0: return
    
    avg_b = base_total / n
    avg_a = adap_total / n
    improvement = (avg_a - avg_b) / (avg_b + 1e-6) * 100

    print("\n" + "="*60)
    print(f"【学术评测报告 - Spatial Alignment (CoT-based)】")
    print(f"评估样本数: {n}")
    print(f"Baseline 准确度: {avg_b:.4f}")
    print(f"Adaptive 准确度: {avg_a:.4f}")
    print(f"绝对提升 (Absolute Margin): {avg_a - avg_b:+.4f}")
    print(f"相对提升 (Relative Boost): {improvement:+.2f}%")
    print("="*60)

    # 保存极其详细的 JSON（包含 LLaVA 的评语，方便你写论文时挑选 Case）
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()