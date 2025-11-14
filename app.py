import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr

from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler
)

from transformers import CLIPModel, CLIPProcessor
pip uninstall -y torch torchvision diffusers transformers accelerate xformers


# ---------------------------
# Load ControlNet + SDXL
# ---------------------------

MODEL_PATH = "/content/drive/MyDrive/SDXL_LoRA_Output"

controlnet = ControlNetModel.from_pretrained(
    f"{MODEL_PATH}/controlnet",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    MODEL_PATH,
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()



# ---------------------------
# Load CLIP for Style Detection
# ---------------------------

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

STYLE_LABELS = ["modern", "industrial", "boho", "luxury", "minimalist", "scandinavian"]


def get_style_type(style_img: Image.Image):
    """Detect interior style (modern, industrial, minimalist, etc.)"""
    inputs = clip_processor(
        text=STYLE_LABELS,
        images=style_img,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    outputs = clip_model(**inputs)
    best = outputs.logits_per_image.argmax().item()
    return STYLE_LABELS[best]



# ---------------------------
# Canny Edge for ControlNet
# ---------------------------

def get_canny(image: Image.Image):
    img = np.array(image)
    img = cv2.resize(img, (1024, 1024))
    edges = cv2.Canny(img, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)



# ---------------------------
# Style Transfer Function
# ---------------------------

def stylize_interior(content_img, style_img):

    content = Image.fromarray(content_img).convert("RGB")
    style = Image.fromarray(style_img).convert("RGB")

    # Detect style
    detected_style = get_style_type(style)
    print("Detected Style:", detected_style)

    # Canny map
    control_map = get_canny(content)

    # Generate styled output
    result = pipe(
        prompt=f"{detected_style} interior design, photorealistic, 8k, highly detailed",
        image=content,
        control_image=control_map,
        num_inference_steps=35,
        guidance_scale=8.0,
    ).images[0]

    return result



# ---------------------------
# GRADIO UI
# ---------------------------

with gr.Blocks() as demo:
    gr.Markdown("<h1><center>AI Interior Style Transfer</center></h1>")
    gr.Markdown("Upload a room image + a style image → Get a new styled interior")

    with gr.Row():
        content_input = gr.Image(label="Upload Room Image", type="numpy")
        style_input = gr.Image(label="Upload Style Reference Image", type="numpy")

    output = gr.Image(label="Stylized Interior Output")

    run_btn = gr.Button("Generate")

    run_btn.click(
        fn=stylize_interior,
        inputs=[content_input, style_input],
        outputs=[output]
    )

demo.launch()

