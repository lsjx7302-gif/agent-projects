# -*- coding: utf-8 -*-
"""Probe DashScope models with both keys; print status only (never the key)."""
import json
import os
import time
import urllib.error
import urllib.request

URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
KEYS = {
    "DASHSCOPE": os.environ.get("DASHSCOPE_API_KEY"),
    "OPENAI": os.environ.get("OPENAI_API_KEY"),
}
MODELS = [
    "qwen-flash", "qwen-turbo", "qwen-plus", "qwen-max",
    "qwen3.8-flash", "qwen2.5-7b-instruct", "qwen3-8b",
    "qwen-turbo-latest", "qwen-plus-latest",
]

for key_name, key in KEYS.items():
    if not key:
        print(f"{key_name}: not set")
        continue
    for model in MODELS:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "回复OK"}],
            "temperature": 0,
            "max_tokens": 5,
        }
        req = urllib.request.Request(
            URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            print(f"OK   {key_name:9s} {model:24s} {data['choices'][0]['message']['content']!r}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            code = ""
            try:
                code = json.loads(body).get("error", {}).get("code", "")
            except Exception:
                pass
            print(f"ERR  {key_name:9s} {model:24s} {e.code} {code}")
        except Exception as e:
            print(f"ERR  {key_name:9s} {model:24s} {type(e).__name__}")
        time.sleep(2)
