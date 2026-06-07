import os, base64, io, torch
from flask import Flask, request, jsonify
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import traceback

app = Flask(__name__)
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

print("Loading LLaVA-Next (4-bit quantized)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto"
)
print("LLaVA-Next ready!")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    try:
        data = request.json
        messages = data.get("messages", [])

        text_parts = []
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        text_parts.append(item["text"])
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                            images.append(img)
            else:
                text_parts.append(str(content))

        prompt_text = " ".join(text_parts)

        if images:
            conversation = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt_text}]}]
        else:
            conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=data.get("max_tokens", 512),
                do_sample=False,
            )

        generated = processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        del inputs, output
        torch.cuda.empty_cache()

        return jsonify({
            "choices": [{"message": {"content": generated, "role": "assistant"}, "finish_reason": "stop"}],
            "model": "llava-next"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("SERVER_PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=False)
