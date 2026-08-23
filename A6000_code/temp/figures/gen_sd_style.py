import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                               torch_dtype=torch.float32, safety_checker=None)
pipe = pipe.to("mps"); pipe.enable_attention_slicing()
prompt = (
    "a clean white background scientific neural network architecture diagram, "
    "colorful rounded rectangle modules connected by arrows, left to right flow, "
    "ocean storm surge forecasting, blue green orange color blocks, flat vector illustration, "
    "minimal, crisp, professional paper figure, no text"
)
neg = "dark background, blurry, messy, distorted, watermark, signature, photo"
for seed in [10, 11, 12]:
    g = torch.Generator("mps").manual_seed(seed)
    img = pipe(prompt, negative_prompt=neg, num_inference_steps=30, guidance_scale=7.5,
               width=768, height=448, generator=g).images[0]
    img.save(f"temp/figures/sd_style_seed{seed}.png")
    print(f"saved sd_style_seed{seed}.png")
