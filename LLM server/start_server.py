"""
Launcher dùng chung cho tất cả MLLM servers.

Cách dùng:
    python start_server.py --model qwen        # mặc định port 8000
    python start_server.py --model llava
    python start_server.py --model qwen --port 8001

Thêm model mới:
    1. Tạo file servers/<tên>_server.py
    2. Thêm entry vào MODEL_REGISTRY bên dưới
"""

import subprocess, time, requests, os, argparse

# ── Registry: thêm model mới vào đây ──────────────────────────────────────────
MODEL_REGISTRY = {
    "qwen": {
        "script": "servers/qwen_server.py",
        "pip":    ["qwen-vl-utils"],
        "log":    "/kaggle/working/qwen_server.log",
    },
    "llava": {
        "script": "servers/llava_server.py",
        "pip":    [],
        "log":    "/kaggle/working/llava_server.log",
    },
    # Ví dụ thêm InternVL sau này:
    # "internvl": {
    #     "script": "servers/internvl_server.py",
    #     "pip":    ["timm"],
    #     "log":    "/kaggle/working/internvl_server.log",
    # },
}
# ──────────────────────────────────────────────────────────────────────────────

HF_CACHE_DIR = "/kaggle/working/hf_cache"
REPO_DIR     = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY.keys(),
                        help="Tên model muốn host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port cho Flask server (mặc định: 8000)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Thời gian chờ tối đa (giây) để server ready")
    return parser.parse_args()

def install_deps(packages: list):
    if not packages:
        return
    pkgs = " ".join(packages)
    print(f"Cài dependencies: {pkgs}")
    os.system(f"pip install -q {pkgs}")

def start_server(script_path: str, port: int, log_path: str) -> subprocess.Popen:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    env = {
        **os.environ,
        "HF_HOME": HF_CACHE_DIR,
        "TRANSFORMERS_CACHE": HF_CACHE_DIR,
        "SERVER_PORT": str(port),
    }
    proc = subprocess.Popen(
        ["python", script_path],
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    return proc

def wait_until_ready(port: int, timeout: int, interval: int = 15):
    elapsed = 0
    while elapsed < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval
        print(f"   ...{elapsed}s", end="\r")
    return False

def main():
    args = parse_args()
    cfg  = MODEL_REGISTRY[args.model]

    script_path = os.path.join(REPO_DIR, cfg["script"])
    if not os.path.exists(script_path):
        print(f"[ERROR] Không tìm thấy script: {script_path}")
        return

    print(f"\n{'='*50}")
    print(f"  Model  : {args.model}")
    print(f"  Port   : {args.port}")
    print(f"  Script : {script_path}")
    print(f"  Log    : {cfg['log']}")
    print(f"{'='*50}\n")

    install_deps(cfg["pip"])

    proc = start_server(script_path, args.port, cfg["log"])
    print(f"Server đang khởi động (PID={proc.pid})...")

    if wait_until_ready(args.port, args.timeout):
        print(f"\n✓ {args.model} server sẵn sàng tại http://localhost:{args.port}")
    else:
        print(f"\n✗ Timeout sau {args.timeout}s! Log cuối:")
        os.system(f"tail -30 {cfg['log']}")

if __name__ == "__main__":
    main()
