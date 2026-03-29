import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from safetensors.torch import load_file as load_sft

# 设置显存碎片优化（非常关键！）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# 导入 FLUX 源码
sys.path.append("/home/mcy/flux/src")
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from flux.util import configs
from flux.sampling import get_schedule, prepare, unpack
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel

class ModelContainer:
    def __init__(self):
        # 修改这里：换到更空的 GPU 2 和 GPU 3
        self.dev_main = torch.device("cuda:0") # GPU 2 跑 Flux (约 24GB-30GB)
        self.dev_text = torch.device("cuda:1") # GPU 3 跑 Text/VAE (约 11GB)
        self.ckpt_path = "/home/mcy/flux/ckpts"
        self.load_models()

    def load_models(self):
        spec = configs["flux-dev"]
        
        # 1. Flux Transformer (约 24GB)
        print(f">>> Loading Flux to {self.dev_main}...")
        self.model = Flux(spec.params).to(self.dev_main).to(torch.bfloat16).eval()
        self.model.load_state_dict(load_sft(os.path.join(self.ckpt_path, "flux1-dev.safetensors")))
        
        # 2. VAE 和 Text Encoders (约 11GB)
        print(f">>> Loading Text Encoders & VAE to {self.dev_text}...")
        self.ae = AutoEncoder(spec.ae_params).to(self.dev_text).to(torch.bfloat16).eval()
        self.ae.load_state_dict(load_sft(os.path.join(self.ckpt_path, "ae.safetensors")))

        self.clip_m = CLIPTextModel.from_pretrained(os.path.join(self.ckpt_path, "text_encoder")).to(self.dev_text, dtype=torch.bfloat16)
        self.clip_t = CLIPTokenizer.from_pretrained(os.path.join(self.ckpt_path, "tokenizer"))
        self.t5_m = T5EncoderModel.from_pretrained(os.path.join(self.ckpt_path, "text_encoder_2")).to(self.dev_text, dtype=torch.bfloat16)
        self.t5_t = T5Tokenizer.from_pretrained(os.path.join(self.ckpt_path, "tokenizer_2"))
    def encode(self, prompt):
        # 修正：分别为 T5 和 CLIP 定义不同的包装行为
        class T5Wrapper:
            def __init__(self, m, t): self.model, self.tokenizer = m, t
            def __call__(self, text):
                toks = self.tokenizer(text, truncation=True, max_length=256, padding="max_length", return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    # T5 需要完整的序列特征: [batch, 256, dim]
                    return self.model(input_ids=toks.input_ids).last_hidden_state

        class CLIPWrapper:
            def __init__(self, m, t): self.model, self.tokenizer = m, t
            def __call__(self, text):
                toks = self.tokenizer(text, truncation=True, max_length=77, padding="max_length", return_tensors="pt").to(self.model.device)
                with torch.no_grad():
                    # CLIP 在 FLUX 中只需要全局池化特征: [batch, dim]
                    # 如果没有 pooler_output，就取第一个 token (CLS)
                    res = self.model(input_ids=toks.input_ids)
                    return res.pooler_output if hasattr(res, 'pooler_output') else res.last_hidden_state[:, 0]

        # 调用时传入修正后的包装类
        return prepare(t5=T5Wrapper(self.t5_m, self.t5_t), 
                       clip=CLIPWrapper(self.clip_m, self.clip_t), 
                       img=torch.zeros(1, 16, 128, 128), # 这里决定了 4096 个 patch
                       prompt=prompt)
_CONTAINER = None

def run_inference(prompt, method="adaptive", save_path="output.png"):
    global _CONTAINER
    if _CONTAINER is None:
        _CONTAINER = ModelContainer()

    with torch.no_grad():
        # 1. 文本编码阶段 (GPU 5)
        opts = _CONTAINER.encode(prompt)
        uncond_opts = _CONTAINER.encode("")
        
        dev = _CONTAINER.dev_main 
        x = torch.randn(1, 4096, 64, device=dev, dtype=torch.bfloat16)
        timesteps = get_schedule(28, x.shape[1], shift=True)

        # 2. 扩散推理阶段 (GPU 4)
        for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((1,), t_curr, device=dev, dtype=torch.bfloat16)
            
            # Cond pass
            v_c = _CONTAINER.model(
                img=x, 
                img_ids=opts['img_ids'].to(dev), 
                txt=opts['txt'].to(dev), 
                txt_ids=opts['txt_ids'].to(dev), 
                y=opts['vec'].to(dev), 
                timesteps=t_vec, 
                guidance=torch.full((1,), 3.5, device=dev, dtype=torch.bfloat16)
            )
            
            if method == "adaptive":
                # Uncond pass (CFG 核心)
                v_u = _CONTAINER.model(
                    img=x, 
                    img_ids=uncond_opts['img_ids'].to(dev), 
                    txt=uncond_opts['txt'].to(dev), 
                    txt_ids=uncond_opts['txt_ids'].to(dev), 
                    y=uncond_opts['vec'].to(dev), 
                    timesteps=t_vec, 
                    guidance=torch.full((1,), 1.0, device=dev, dtype=torch.bfloat16)
                )
                
                # Adaptive CFG 计算 (就在 GPU 4 完成)
                diff = v_c - v_u
                mag = torch.norm(diff, dim=-1)
                # 将 mag 转换回空间布局进行池化 [1, 1, 64, 64]
                mag_map = F.avg_pool2d(mag.view(1, 1, 64, 64), kernel_size=3, stride=1, padding=1)
                norm_mag = (mag_map - mag_map.min()) / (mag_map.max() - mag_map.min() + 1e-6)
                adaptive_scale = 2.0 + 3.0 * norm_mag
                v_final = v_u + adaptive_scale.view(1, 4096, 1) * (diff / 3.5)
                
                # 释放临时变量
                del v_c, v_u, diff, mag, mag_map
            else:
                v_final = v_c

            x = x + (t_next - t_curr) * v_final

        # 3. 解码阶段 (搬回 GPU 5)
        print(f">>> Decoding on {_CONTAINER.dev_text}...")
        x_vae = unpack(x.to(_CONTAINER.dev_text), 1024, 1024)
        out = _CONTAINER.ae.decode(x_vae)
        
        # 4. 保存与显存清理
        img = (out.clamp(-1, 1) + 1).mul(127.5)[0].permute(1, 2, 0).cpu().byte().numpy()
        Image.fromarray(img).save(save_path)
        
        del x, x_vae, out, opts, uncond_opts
        # 每一张图跑完，手动清理一下，防止 600 张连跑时的显存堆积
        if i % 10 == 0:
            torch.cuda.empty_cache()