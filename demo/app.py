"""
Gradio demo for PaddleOCR-VL fine-tuned on Aksara Jawa (Javanese script).

Runs on HuggingFace Spaces (ZeroGPU). Loads the fine-tuned model via the
transformers interface with trust_remote_code=True, takes an image, returns
the recognised Unicode Aksara Jawa transcription.

Set MODEL_PATH env var to switch between base and fine-tuned models.
Default: setilanaji/PaddleOCR-VL-Aksara-Jawa.
"""

import os

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Default to base PaddleOCR-VL until the fine-tune finishes training.
# Once setilanaji/PaddleOCR-VL-Aksara-Jawa exists, set MODEL_PATH=... in the Space settings.
MODEL_PATH = os.environ.get("MODEL_PATH", "PaddlePaddle/PaddleOCR-VL")
PROMPT = os.environ.get("OCR_PROMPT", "OCR:")

model = None
processor = None


def load_model() -> None:
    global model, processor
    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=True
    )
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = processor.tokenizer.eos_token_id
    print("Model loaded.")


load_model()


@spaces.GPU(duration=90)
def recognize(image: Image.Image | None) -> str:
    if image is None:
        return "⚠️ Upload an image or pick an example below."

    if model.device.type == "cpu" and torch.cuda.is_available():
        model.to("cuda")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

    input_length = inputs["input_ids"].shape[1]
    answer = processor.batch_decode(
        generated[:, input_length:], skip_special_tokens=True
    )[0]
    return answer.strip()


DESCRIPTION = """
# PaddleOCR-VL — Aksara Jawa (Javanese Script) OCR

Transcribes **Aksara Jawa** (ꦲꦏ꧀ꦱꦫꦗꦮ) manuscript and signage images to Unicode.
Fine-tuned from [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) using LoRA on a mixed
synthetic + real Javanese manuscript dataset.

**Model**: [setilanaji/PaddleOCR-VL-Aksara-Jawa](https://huggingface.co/setilanaji/PaddleOCR-VL-Aksara-Jawa) ·
**Code**: [github.com/setilanaji/paddleocr-aksara-jawa](https://github.com/setilanaji/paddleocr-aksara-jawa) ·
**Built for**: PaddleOCR Global Derivative Model Challenge (Hackathon 10th)

Why this matters: Aksara Jawa is the traditional script of over 80 million Javanese speakers, part of the
Indonesian school curriculum in Central/East Java and DIY, and the writing system used in tens of thousands
of manuscripts held by PNRI, EAP, and Dreamsea — but no publicly available OCR model exists for it.
"""


with gr.Blocks(title="PaddleOCR-VL — Aksara Jawa OCR", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Upload Aksara Jawa image", height=360)
            run_btn = gr.Button("Transcribe ꦲ", variant="primary")

            examples_dir = os.path.join(os.path.dirname(__file__), "examples")
            example_files = sorted(
                os.path.join(examples_dir, f)
                for f in os.listdir(examples_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ) if os.path.isdir(examples_dir) else []
            if example_files:
                gr.Examples(
                    examples=[[f] for f in example_files],
                    inputs=image_in,
                    label="Example Images",
                )

        with gr.Column():
            text_out = gr.Textbox(
                label="Unicode Aksara Jawa Transcription",
                placeholder="Transkripsi Unicode Aksara Jawa akan muncul di sini...",
                lines=12,
                max_lines=20,
                show_copy_button=True,
            )

    run_btn.click(fn=recognize, inputs=image_in, outputs=text_out)
    image_in.change(fn=recognize, inputs=image_in, outputs=text_out)


if __name__ == "__main__":
    demo.launch()
