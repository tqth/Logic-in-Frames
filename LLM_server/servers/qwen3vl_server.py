"""
Flask server host Qwen3-VL-4B-Instruct bang HuggingFace transformers.
Duoc goi boi start_server.py
"""
import os
import io
import base64
import traceback

import torch
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

app = Flask(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"

print(f"Loading {MODEL_ID} ...")

quantization_config = None
# if LOAD_IN_4BIT:
#     # 4B fit thoai mai o fp16/bf16 tren T4 (~8GB), chi bat 4-bit neu ban can
#     # chia se VRAM voi tien trinh khac (vd PaddleOCR GPU chay song song).
#     from transformers import BitsAndBytesConfig
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
#     print("Bat 4-bit quantization (LOAD_IN_4BIT=1).")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype="auto",
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="sdpa",
)
print(f"{MODEL_ID} ready!")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_ID})


def _b64_to_pil(data_url_or_b64: str) -> Image.Image:
    """Nhan chuoi base64 thuan hoac data URL (data:image/...;base64,xxx)."""
    if data_url_or_b64.startswith("data:image"):
        b64 = data_url_or_b64.split(",", 1)[1]
    else:
        b64 = data_url_or_b64
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _openai_messages_to_qwen(messages):
    """
    Convert message dang OpenAI-compatible (content la list voi type
    text/image_url) sang dinh dang content ma Qwen3-VL hieu (image la
    PIL.Image truc tiep).
    """
    qwen_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            qwen_content = []
            for item in content:
                if item.get("type") == "text":
                    qwen_content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    url = item["image_url"]["url"]
                    img = _b64_to_pil(url)
                    qwen_content.append({"type": "image", "image": img})
            qwen_messages.append({"role": role, "content": qwen_content})
        else:
            qwen_messages.append({"role": role, "content": [{"type": "text", "text": str(content)}]})
    return qwen_messages


@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    try:
        data = request.json
        messages = data.get("messages", [])
        max_new_tokens = data.get("max_tokens", 512)

        qwen_messages = _openai_messages_to_qwen(messages)

        inputs = processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        # tranh loi token_type_ids voi mot so ban transformers
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        generated_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs, generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({
            "choices": [{"message": {"content": generated_text, "role": "assistant"}, "finish_reason": "stop"}],
            "model": MODEL_ID,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=False)