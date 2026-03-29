import torch
import os
import json
import re
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig 

# --- 配置路径 ---
BASE_DIR = "/home/mcy/flux/exp_results/T2I_CompBench_complex/baseline/samples"
ADAP_DIR = "/home/mcy/flux/exp_results/T2I_CompBench_complex/adaptive/samples"
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OUTPUT_JSON = "complex_eval_paper_summary.json"

print(">>> 正在加载 LLaVA-1.5-7B 模型 (采用论文标准 CoT 评测 - Complex Scene)...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=quantization_config,
    torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
)

def get_score_from_llava(image_path, prompt_text):
    # 【Complex Scene 专属量表】
    instruction = (
        f"USER: <image>\nYou are an expert evaluator for text-to-image models. "
        f"Evaluate the COMPOSITIONAL INTEGRITY based on this complex prompt: '{prompt_text}'.\n\n"
        "Rubric:\n"
        "3 - Perfect: ALL mentioned objects, their specific attributes, and their relationships are perfectly present.\n"
        "2 - Slight Omission/Ambiguity: Most objects are present, but a minor background object is missing, or there is a slight mix-up in minor attributes.\n"
        "1 - Severe Confusion: Key objects are merged together (frankenstein-ing), major attribute leakage occurs, or significant portions of the prompt are ignored.\n"
        "0 - Total Failure: Multiple primary objects are missing, or the image completely fails to capture the core subject.\n\n"
        "Step 1: List the objects found in the image and check if they match the prompt.\n"
        "Step 2: Provide a final score (0, 1, 2, or 3) based on the rubric.\n"
        "Format your response exactly as follows:\n"
        "Reasoning: [your analysis]\n"
        "Score: [your score]\nASSISTANT:"
    )
    
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(text=instruction, images=raw_image, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=150, temperature=0.2, do_sample=True)
        res_text = processor.decode(output[0], skip_special_tokens=True)
        ans = res_text.split("ASSISTANT:")[-1].strip()
        
        match = re.search(r"Score:\s*([0-3])", ans, re.IGNORECASE)
        raw_score = int(match.group(1)) if match else (int(re.findall(r"\b[0-3]\b", ans)[-1]) if re.findall(r"\b[0-3]\b", ans) else 1)
        return raw_score / 3.0, ans
    except Exception as e:
        print(f"Error evaluating {image_path}: {e}")
        return 0.33, "Error during inference"

def main():
    print(f">>> 检查路径: {BASE_DIR}")
    base_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith('.png')])
    if not base_files: return print("!!! 错误：Baseline 文件夹中没有找到 .png 图片。")

    results = []
    base_total, adap_total = 0, 0
    
    print(f">>> 开始对 {len(base_files)} 个样本进行 Complex 深度交叉打分...")
    for filename in tqdm(base_files):
        img_id = filename.split('_')[0]
        prompt = " ".join(filename.split('_')[1:]).rsplit('.', 1)[0].replace('_', ' ')
        
        path_b, path_a = os.path.join(BASE_DIR, filename), os.path.join(ADAP_DIR, filename)
        if not os.path.exists(path_a): continue
        
        score_b, reason_b = get_score_from_llava(path_b, prompt)
        score_a, reason_a = get_score_from_llava(path_a, prompt)
        
        base_total += score_b
        adap_total += score_a
        results.append({
            "id": img_id, "prompt": prompt, "margin": score_a - score_b,
            "baseline": {"score": score_b, "reasoning": reason_b},
            "adaptive": {"score": score_a, "reasoning": reason_a}
        })

    n = len(results)
    avg_b, avg_a = base_total / n, adap_total / n
    print(f"\n【学术评测报告 - Complex Scene (CoT-based)】")
    print(f"Baseline 准确度: {avg_b:.4f} | Adaptive 准确度: {avg_a:.4f} | 绝对提升: {avg_a - avg_b:+.4f}")
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()