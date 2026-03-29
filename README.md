1. 推理与生图
flux_pt_acfg.py: 核心推理脚本。

run_t2i_compbench.py: 文本生成图像（T2I）脚本，用于在 CompBench 基准上进行生图测试。

test_pipeline.py: 推理测试脚本

2. 评分 (基于 LLaVA)
这些脚本利用 LLaVA 模型对生成的图片进行多维度打分：

llava_eval_color.py: 针对 Color属性的准确性进行评分。

llava_eval_complex.py: 针对 Complex 的遵循度进行评分。

llava_eval_spatial.py: 针对 Spatial 的合理性进行评分。
基于BILP_vqa的评分由于权限问题无法上传

3. 结果
success_compare.py: 整理成功的案例。

fault_compare.py: 整理生成失败或存在缺陷的案例。
