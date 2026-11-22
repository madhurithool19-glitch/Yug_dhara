import os
import time
import urllib.parse
import urllib.request

os.makedirs("materials", exist_ok=True)

with open("video-factory/prompts.txt", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

pipe = None
for i, line in enumerate(lines[:6]):
    tag, _, prompt = line.partition(":")
    tag = tag.strip().lower()
    prompt = prompt.strip()
    out = f"materials/img_{i}.jpg"
    if os.path.exists(out) and os.path.getsize(out) > 20000:
        print(f"scene {i}: already present, skipping", flush=True)
        continue
    if tag == "turbo":
        try:
            if pipe is None:
                import torch
                from diffusers import AutoPipelineForText2Image

                pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sd-turbo", torch_dtype=torch.float32
                )
                pipe.to("cpu")
                print("sd-turbo model loaded", flush=True)
            img = pipe(
                prompt=prompt,
                num_inference_steps=2,
                guidance_scale=0.0,
                width=512,
                height=768,
            ).images[0]
            img.save(out)
            print(f"scene {i}: turbo done", flush=True)
        except Exception as e:
            print(f"scene {i}: turbo FAILED {e}", flush=True)
    else:
        enc = urllib.parse.quote(prompt)
        ok = False
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"https://image.pollinations.ai/prompt/{enc}?width=1080&height=1920&seed={100 + i}&nologo=true",
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                data = urllib.request.urlopen(req, timeout=150).read()
                if len(data) > 20000:
                    with open(out, "wb") as fh:
                        fh.write(data)
                    ok = True
                    break
            except Exception as e:
                print(f"scene {i}: poll attempt {attempt} failed {e}", flush=True)
            time.sleep(15)
        print(f"scene {i}: poll {'done' if ok else 'FAILED'}", flush=True)
        time.sleep(10)

print("gen_scenes finished", flush=True)
