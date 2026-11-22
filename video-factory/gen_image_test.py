import sys
import time

import torch
from diffusers import AutoPipelineForText2Image

PROMPT = (
    "young Indian man running a small tea stall outside a small town railway "
    "station in Maharashtra, morning sunlight, steam rising from a kettle, "
    "cinematic, photorealistic, vertical portrait"
)

prompt = sys.argv[1] if len(sys.argv) > 1 else PROMPT

t0 = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float32
)
pipe.to("cpu")
print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

t1 = time.time()
image = pipe(
    prompt=prompt,
    num_inference_steps=2,
    guidance_scale=0.0,
    width=512,
    height=768,
).images[0]
image.save("test_image.png")
print(f"image generated in {time.time() - t1:.1f}s, saved test_image.png", flush=True)
