# LLM Server

Host các Multimodal LLM dưới dạng Flask API tương thích OpenAI format, chạy trên Kaggle (GPU T4/P100).

## Cấu trúc

```
LLM_server/
├── servers/
│   ├── qwen_server.py      # Qwen2.5-VL-7B-Instruct
│   └── llava_server.py     # LLaVA-v1.6-Mistral-7B
├── start_server.py         # Launcher dùng chung
├── requirements.txt
└── README.md
```

## Cách dùng

```bash
# Host Qwen2.5-VL (port mặc định 8000)
python start_server.py --model qwen

# Host LLaVA
python start_server.py --model llava

# Tuỳ chỉnh port và timeout
python start_server.py --model qwen --port 8001 --timeout 900
```

## API

Tất cả server đều expose endpoint tương thích OpenAI:

```
GET  /health
POST /v1/chat/completions
```

Ví dụ gọi với ảnh:

```python
import requests, base64

with open("image.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/v1/chat/completions", json={
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Describe this image."}
        ]
    }],
    "max_tokens": 512
})
print(resp.json()["choices"][0]["message"]["content"])
```

## Thêm model mới

1. Tạo `servers/<tên>_server.py` (copy template từ file có sẵn, thay processor/model class)
2. Thêm entry vào `MODEL_REGISTRY` trong `start_server.py`:

```python
"internvl": {
    "script": "servers/internvl_server.py",
    "pip":    ["timm"],
    "log":    "/kaggle/working/internvl_server.log",
},
```

3. Chạy: `python start_server.py --model internvl`
