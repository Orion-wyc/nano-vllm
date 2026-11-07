import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def resolve_model_path():
    env_path = os.environ.get("NANO_VLLM_MODEL")
    candidates = [env_path] if env_path else []
    candidates += [
        "/home/developer/models/Qwen3.5-0.8B",
        "/home/developer/models/models/Qwen--Qwen3.5-0.8B/snapshots/master",
        "/mnt/workspace/models/Qwen3.5-0.8B",
    ]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.expanduser(candidate)):
            return os.path.abspath(os.path.expanduser(candidate))
    raise FileNotFoundError("model directory not found; set NANO_VLLM_MODEL")


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def read_log(path):
    try:
        return Path(path).read_text(errors="ignore")[-8000:]
    except OSError:
        return ""


def wait_for_health(client, base_url, proc, log_path, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited with code {proc.returncode}\n{read_log(log_path)}"
            )
        try:
            response = client.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                payload = response.json()
                if payload.get("status") == "ok":
                    return payload
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy: {last_error}\n{read_log(log_path)}")


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main():
    path = resolve_model_path()
    prompt = os.environ.get("NANO_VLLM_PROMPT", "Hello, introduce yourself in one sentence.")
    device_id = int(os.environ.get("NANO_VLLM_DEVICE_ID", "0"))
    timeout = float(os.environ.get("NANO_VLLM_API_TIMEOUT", "600"))
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        prefix="nano_vllm_api_",
        suffix=".log",
        delete=False,
    )
    log_path = log_file.name
    log_file.close()
    log_handle = open(log_path, "w")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "nanovllm.serving.api_server",
        "--model",
        path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device-id",
        str(device_id),
        "--max-model-len",
        "256",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "256",
        "--enforce-eager",
    ]
    proc = None
    client = None
    try:
        print(f"model={path}")
        print(f"server={base_url}")
        print(f"command={' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client = httpx.Client(
            timeout=httpx.Timeout(300.0, connect=10.0),
            trust_env=False,
        )
        health = wait_for_health(client, base_url, proc, log_path, timeout)
        print(f"health={health}")

        model_name = path
        models_response = client.get(f"{base_url}/v1/models")
        print(f"models_status={models_response.status_code}")
        if models_response.status_code == 200:
            models = models_response.json().get("data") or []
            if models and models[0].get("id"):
                model_name = models[0]["id"]
            print(f"models_response={json.dumps(models_response.json(), ensure_ascii=False)[:1000]}")
        else:
            print(f"models_response={models_response.text[:1000]}")

        payload = {
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 1e-5,
            "stream": False,
        }
        response = client.post(f"{base_url}/v1/completions", json=payload)
        print(f"completion_status={response.status_code}")
        print(f"completion_response={response.text[:2000]}")
        if response.status_code != 200:
            print("ERROR: non-stream completion failed", file=sys.stderr)
            return 1
        body = response.json()
        choices = body.get("choices") or []
        text = choices[0].get("text", "") if choices else ""
        if not text:
            print("ERROR: non-stream completion returned empty text", file=sys.stderr)
            return 1

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        data_events = 0
        done = False
        sample_chunks = []
        with client.stream("POST", f"{base_url}/v1/completions", json=stream_payload) as stream:
            print(f"stream_status={stream.status_code}")
            if stream.status_code != 200:
                print(f"ERROR: stream completion failed: {stream.read()[:2000]}", file=sys.stderr)
                return 1
            for line in stream.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    continue
                data_events += 1
                if len(sample_chunks) < 3:
                    sample_chunks.append(data)
        print(f"stream_data_events={data_events}")
        print(f"stream_done={done}")
        print(f"stream_samples={sample_chunks}")
        if data_events < 1:
            print("ERROR: no stream data events received", file=sys.stderr)
            return 1
        if not done:
            print("ERROR: stream did not end with [DONE]", file=sys.stderr)
            return 1

        chat_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 1e-5,
            "stream": False,
        }
        try:
            chat_response = client.post(f"{base_url}/v1/chat/completions", json=chat_payload)
            print(f"chat_status={chat_response.status_code}")
            print(f"chat_response={chat_response.text[:2000]}")
            if chat_response.status_code != 200:
                print("WARNING: optional chat completion failed", file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: optional chat completion error: {exc}", file=sys.stderr)

        print("API server verification passed")
        return 0
    finally:
        if client is not None:
            client.close()
        stop_process(proc)
        log_handle.close()
        if proc is not None and proc.returncode not in (0, None, -15):
            print(f"server_log:\n{read_log(log_path)}", file=sys.stderr)
        try:
            os.unlink(log_path)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
