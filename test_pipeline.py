import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from safetensors.torch import load_file as load_sft
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel

# 导入路径
sys.path.append("/home/mcy/flux/src")
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from flux.util import configs
from flux.sampling import get_schedule, prepare, unpack

# --- 编码器包装类 ---
class T5EncoderWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer, max_length=256):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
    def forward(self, text):
        tokens = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt").to(self.model.device)
        return self.model(input_ids=tokens.input_ids).last_hidden_state

class CLIPEncoderWrapper(torch.nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    def forward(self, text):
        tokens = self.tokenizer(text, truncation=True, max_length=77, padding="max_length", return_tensors="pt").to(self.model.device)
        return self.model(input_ids=tokens.input_ids).pooler_output

def save_discrepancy_heatmap(dist_map, step, t, output_path, h, w, title_prefix="Discrepancy"):
    grid_h, grid_w = h // 16, w // 16
    heatmap = dist_map.view(grid_h, grid_w).detach().cpu().float().numpy()
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='magma')
    plt.colorbar(label='L2 Norm')
    plt.title(f"{title_prefix} | Step {step:02d} | T {t:.3f}")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

@torch.no_grad()
def main():
    device_flux, device_text = torch.device("cuda:6"), torch.device("cuda:7")
    base_path = "/home/mcy/flux"
    ckpt_path, output_dir, tensor_dir = os.path.join(base_path, "ckpts"), os.path.join(base_path, "vis_discrepancy_results"), os.path.join(base_path, "pred_tensors")
    os.makedirs(output_dir, exist_ok=True); os.makedirs(tensor_dir, exist_ok=True)
    
    # --- 新的测试 Prompt ---
    prompt = "A cat playing in the snow"
    height, width, num_steps, guidance_scale, seed = 1024, 1024, 28, 3.5, 42

    print(f"Loading models for prompt: {prompt}")
    spec = configs["flux-dev"]
    model = Flux(spec.params).to(device_flux).to(torch.bfloat16)
    model.load_state_dict(load_sft(os.path.join(ckpt_path, "flux1-dev.safetensors")))
    ae = AutoEncoder(spec.ae_params).to(device_flux).to(torch.bfloat16)
    ae.load_state_dict(load_sft(os.path.join(ckpt_path, "ae.safetensors")))

    clip_m = CLIPTextModel.from_pretrained(os.path.join(ckpt_path, "text_encoder")).to(device_text, dtype=torch.bfloat16)
    clip = CLIPEncoderWrapper(clip_m, CLIPTokenizer.from_pretrained(os.path.join(ckpt_path, "tokenizer")))
    t5_m = T5EncoderModel.from_pretrained(os.path.join(ckpt_path, "text_encoder_2")).to(device_text, dtype=torch.bfloat16)
    t5 = T5EncoderWrapper(t5_m, T5Tokenizer.from_pretrained(os.path.join(ckpt_path, "tokenizer_2")))

    torch.manual_seed(seed)
    opts = prepare(t5=t5, clip=clip, img=torch.zeros(1, 16, height//8, width//8), prompt=prompt)
    uncond_opts = prepare(t5=t5, clip=clip, img=torch.zeros(1, 16, height//8, width//8), prompt="")
    x = torch.randn(1, (height // 16) * (width // 16), 16 * 2 * 2, device=device_flux, dtype=torch.bfloat16)
    timesteps = get_schedule(num_steps, x.shape[1], shift=True)

    for i, (t_curr, t_next) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((1,), t_curr, device=device_flux, dtype=torch.bfloat16)
        def get_v(od, g):
            return model(img=x, img_ids=od['img_ids'].to(device_flux), txt=od['txt'].to(device_flux), txt_ids=od['txt_ids'].to(device_flux), y=od['vec'].to(device_flux), timesteps=t_vec, guidance=torch.full((1,), g, device=device_flux, dtype=torch.bfloat16))

        v_cond, v_uncond = get_v(opts, guidance_scale), get_v(uncond_opts, 1.0)

        # 保存二进制 Tensor (师兄要求的任务)
        torch.save(v_cond.detach().cpu().half(), os.path.join(tensor_dir, f"v_cond_step_{i:02d}.pt"))
        torch.save(v_uncond.detach().cpu().half(), os.path.join(tensor_dir, f"v_uncond_step_{i:02d}.pt"))

        # 保存 Discrepancy 热力图 (肉眼观察用)
        save_discrepancy_heatmap(torch.norm(v_cond - v_uncond, dim=-1).squeeze(0), i, t_curr, os.path.join(output_dir, f"discrepancy_step_{i:02d}.png"), height, width)

        x = x + (t_next - t_curr) * v_cond
        print(f"Step {i+1}/{num_steps} | Data Saved", end='\r')

    Image.fromarray((ae.decode(unpack(x, height, width)).clamp(-1, 1) + 1).mul(127.5)[0].permute(1, 2, 0).cpu().byte().numpy()).save(os.path.join(base_path, "final_output.png"))
    print("\nSuccess!")

if __name__ == "__main__":
    main()