import torch
from diffusers import StableDiffusionPipeline
import os

seeds = [1, 42, 31315, 2702, 55555]

prompts = ["mystyle, 2D flat, A classic cocktail is placed alongside a napkin",
           "mystyle, 2D flat, a teddy dressed in a blue suit is cooking a gourmet meal"]

model_ft_path = 'model/2dflat'
res_dir = 'results_' + os.path.split(model_ft_path)[-1]
os.makedirs(res_dir, exist_ok=True)

pipe_ft = StableDiffusionPipeline.from_pretrained(model_ft_path, torch_dtype=torch.float16)
pipe_ft.to("cuda")

for prompt in prompts:
    for seed in seeds:
        g = torch.Generator('cuda').manual_seed(seed)
        image_ft = pipe_ft(prompt=prompt, generator=g).images[0]
        image_ft.save(os.path.join(res_dir, prompt.replace(' ', '')+ '_FT' +f"_seed{seed}.png"))

model_sd_path = "model/stable-diffiusion-v1-5"
pipe_sd = StableDiffusionPipeline.from_pretrained(model_sd_path, torch_dtype=torch.float16)
pipe_sd.to("cuda")

for prompt in prompts:
    for seed in seeds:
        g = torch.Generator('cuda').manual_seed(seed)
        image_sd = pipe_sd(prompt=prompt, generator=g).images[0]
        image_sd.save(os.path.join(res_dir, prompt.replace(' ', '')+ '_SD' +f"_seed{seed}.png"))