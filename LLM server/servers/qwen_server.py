import os, base64, io, torch
from flask import Flask, request, jsonify
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import traceback

app = Flask(__name__)
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Loading Qwen2.5-VL (4-bit quantized)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    # Giới hạn visual tokens per frame — đây là chìa khoá tránh OOM
    min_pixels=224 * 224,
    max_pixels=336 * 336,
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    device_map="auto",
    attn_implementation="sdpa",  # giảm peak memory trong attention
)
print("Qwen2.5-VL ready!")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    try:
        data = request.json
        messages = data.get("messages", [])

        # Parse messages — tách text và images từ OpenAI-compatible format
        # Qwen2.5-VL dùng message format riêng nên cần convert
        qwen_content = []
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        qwen_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            b64 = url.split(",", 1)[1]
                            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                            images.append(img)
                            # Qwen2.5-VL nhận image object trực tiếp trong content
                            qwen_content.append({
                                "type": "image",
                                "image": img,
                                "min_pixels": 224 * 224,
                                "max_pixels": 336 * 336,
                            })
            else:
                qwen_content.append({"type": "text", "text": str(content)})

        qwen_messages = [{"role": "user", "content": qwen_content}]

        # apply_chat_template → text prompt
        prompt = processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # process_vision_info trích xuất image/video tensors từ messages
        image_inputs, video_inputs = process_vision_info(qwen_messages)

        inputs = processor(
            text=[prompt],
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=data.get("max_tokens", 512),
                do_sample=False,
            )

        # Chỉ decode phần generated, bỏ phần prompt
        generated = processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Cleanup để tránh VRAM leak qua nhiều request
        del inputs, output
        torch.cuda.empty_cache()

        return jsonify({
            "choices": [{"message": {"content": generated, "role": "assistant"}, "finish_reason": "stop"}],
            "model": "qwen2.5-vl"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("LLAVA_PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=False)
