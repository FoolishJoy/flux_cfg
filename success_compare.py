import os
import json
from PIL import Image
from fpdf import FPDF
from tqdm import tqdm

# ================= 配置区 =================
SCORE_SUM_ROOT = "/home/mcy/flux/exp_results/score_sum"
IMAGE_ROOT = "/home/mcy/flux/exp_results"
DATASET_ROOT = "/home/mcy/flux/benchmarks/T2I-CompBench/examples/dataset"
CATEGORIES = ["color", "complex", "spatial"]
OUTPUT_ROOT = "/home/mcy/flux/exp_results/Success_Reports_PDF"
MAX_CASES = 20  
# ==========================================

class PDFReport(FPDF):
    def header(self):
        # 修复警告：Arial 换成 helvetica
        self.set_font("helvetica", "B", 15)
        # 修复警告：ln=True 换成 new_x/new_y
        self.cell(0, 10, self.report_title, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(5)

def load_full_prompts(category):
    """针对无 ID 文本文件的加载器：直接使用行号作为 ID 映射"""
    path = os.path.join(DATASET_ROOT, f"{category}.txt")
    p_map = {}
    
    if not os.path.exists(path):
        print(f"!!! 警告：找不到数据集文件 {path}")
        return None
        
    print(f">>> 正在通过行号索引加载数据集: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 使用 enumerate 自动生成从 0 开始的行号 ID
            for idx, line in enumerate(f):
                line = line.strip()
                if not line: continue # 跳过空行
                
                # 建立映射：{0: "The black chair...", 1: "The round clock...", ...}
                p_map[idx] = line
                
        print(f">>> 成功加载 {len(p_map)} 条 Prompt。")
        # 调试打印：检查 144 号是否对得上
        if 144 in p_map:
            print(f">>> [Debug] ID 144 对应的完整 Prompt: {p_map[144][:50]}...")
            
        return p_map
    except Exception as e:
        print(f"!!! 读取文件失败: {e}")
        return None
# 【修复名称一致性】
def create_sbs_image(base_p, adap_p, temp_p):
    try:
        img_b = Image.open(base_p).convert("RGB")
        img_a = Image.open(adap_p).convert("RGB")
        w, h = img_b.size
        # 适当 resize 减小 PDF 体积
        new_w, new_h = 512, 512
        img_b = img_b.resize((new_w, new_h), Image.LANCZOS)
        img_a = img_a.resize((new_w, new_h), Image.LANCZOS)
        
        merged = Image.new('RGB', (new_w * 2, new_h))
        merged.paste(img_b, (0, 0))
        merged.paste(img_a, (new_w, 0))
        merged.save(temp_p, "JPEG", quality=80)
        return True
    except:
        return False

def generate_category_pdf(category):
    print(f"\n>>> 正在分析 [{category.upper()}] 并生成 PDF...")
    prompt_map = load_full_prompts(category)
    os.makedirs(os.path.join(OUTPUT_ROOT, "temp"), exist_ok=True)
    
    score_dir = os.path.join(SCORE_SUM_ROOT, category)
    try:
        base_scores = json.load(open(os.path.join(score_dir, "baseline.json")))
        adap_scores = json.load(open(os.path.join(score_dir, "adaptive.json")))
    except:
        print(f"   [错误] 找不到 {category} 的分数 JSON")
        return

    successes = []
    for fname in base_scores:
        if fname in adap_scores:
            margin = float(adap_scores[fname]) - float(base_scores[fname])
            if margin > 0:
                img_id = int(fname.split('_')[0])
                successes.append({
                    "id": img_id, "filename": fname, "margin": margin,
                    "base": base_scores[fname], "adap": adap_scores[fname],
                    "prompt": prompt_map.get(img_id, "Unknown Prompt") if prompt_map else "Unknown"
                })
    
    successes.sort(key=lambda x: x['margin'], reverse=True)
    successes = successes[:MAX_CASES]

    if not successes:
        print(f"   [提示] {category} 类别下没有发现提升的案例。")
        return

    pdf = PDFReport()
    pdf.report_title = f"Success Analysis: {category.upper()} (Adaptive > Baseline)"
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    img_cat_root = os.path.join(IMAGE_ROOT, f"T2I_CompBench_{category}")

    for i, case in enumerate(tqdm(successes)):
        # 写入文字信息
        pdf.set_font("helvetica", "B", 11)
        pdf.multi_cell(0, 8, f"Case {i+1} [ID: {case['id']}]: {case['prompt']}")
        
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(0, 128, 0) 
        # 修复警告：ln=True 替换
        pdf.cell(0, 6, f"Improvement: Baseline {case['base']:.2f} -> Adaptive {case['adap']:.2f} (Margin: +{case['margin']:.2f})", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0) 

        base_p = os.path.join(img_cat_root, "baseline", "samples", case['filename'])
        adap_p = os.path.join(img_cat_root, "adaptive", "samples", case['filename'])
        temp_p = os.path.join(OUTPUT_ROOT, "temp", f"succ_{category}_{i}.jpg")
        
        # 调用已经统一定义的函数
        if create_sbs_image(base_p, adap_p, temp_p):
            pdf.set_font("helvetica", "I", 8)
            pdf.cell(95, 5, "Baseline (Left)", align="C")
            pdf.cell(95, 5, "Adaptive (Right)", align="C", new_x="LMARGIN", new_y="NEXT")
            pdf.image(temp_p, x=10, w=190)
            pdf.ln(10) 

    save_path = os.path.join(OUTPUT_ROOT, f"{category}_success_report.pdf")
    pdf.output(save_path)
    print(f"✅ PDF 已生成: {save_path}")

def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for cat in CATEGORIES:
        generate_category_pdf(cat)
    
    import shutil
    temp_dir = os.path.join(OUTPUT_ROOT, "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print(f"\n🚀 所有正面报告已生成！")

if __name__ == "__main__":
    main()