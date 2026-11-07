import concurrent.futures
import json
import os
import shutil
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

from nanovllm import SamplingParams
from nanovllm.engine.llm_engine import LLMEngine


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


def find_router_bin():
    env_path = os.environ.get("VLLM_ROUTER_BIN")
    if env_path:
        return env_path
    found = shutil.which("vllm-router")
    if found:
        return found
    candidates = [
        REPO_ROOT.parent / "vllm-router" / "target" / "release" / "vllm-router",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(
        "vllm-router not found; set VLLM_ROUTER_BIN or install the wheel from "
        f"{REPO_ROOT.parent / 'vllm-router' / 'dist'}"
    )


def read_log(path):
    try:
        return Path(path).read_text(errors="ignore")[-8000:]
    except OSError:
        return ""


def wait_http(client, url, proc, log_path, timeout, name):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{name} exited with code {proc.returncode}\n{read_log(log_path)}"
            )
        try:
            response = client.get(url, timeout=2.0)
            if response.status_code // 100 == 2:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"{name} did not become healthy: {last_error}\n{read_log(log_path)}")


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def worker_command(path, role, port, device, worker_id, model_args):
    return [
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
        str(device),
        "--role",
        role,
        "--worker-id",
        worker_id,
        "--advertise-url",
        f"http://127.0.0.1:{port}",
        "--enforce-eager",
        *model_args,
    ]


def direct_generate(path, prompts, max_tokens, device, engine_kwargs):
    engine = LLMEngine(path, device_id=device, **engine_kwargs)
    try:
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        texts = []
        for prompt in prompts:
            output = engine.generate([prompt], params, use_tqdm=False)[0]
            texts.append(
                engine.tokenizer.decode(
                    output["token_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return texts
    finally:
        engine.exit()


def main():
    path = resolve_model_path()
    router_bin = find_router_bin()
    prefill_device = int(os.environ.get("NANO_VLLM_PREFILL_DEVICE", "0"))
    decode_device = int(os.environ.get("NANO_VLLM_DECODE_DEVICE", "1"))
    max_tokens = int(os.environ.get("NANO_VLLM_MAX_TOKENS", "8"))
    timeout = float(os.environ.get("NANO_VLLM_ROUTER_TIMEOUT", "600"))
    gpu_util = os.environ.get("NANO_VLLM_GPU_UTIL", "0.3")
    engine_kwargs = {
        "enforce_eager": True,
        "max_model_len": int(os.environ.get("NANO_VLLM_MAX_MODEL_LEN", "512")),
        "max_num_seqs": int(os.environ.get("NANO_VLLM_MAX_NUM_SEQS", "8")),
        "max_num_batched_tokens": int(os.environ.get("NANO_VLLM_MAX_NUM_BATCHED_TOKENS", "512")),
        "gpu_memory_utilization": float(gpu_util),
    }
    model_args = [
        "--max-model-len",
        str(engine_kwargs["max_model_len"]),
        "--max-num-seqs",
        str(engine_kwargs["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(engine_kwargs["max_num_batched_tokens"]),
        "--gpu-memory-utilization",
        gpu_util,
    ]
    prompts = os.environ.get(
        "NANO_VLLM_PROMPTS",
        "Hello, introduce yourself in one sentence.||What is 1+1?",
    ).split("||")

    prefill_port = free_port()
    decode_port = free_port()
    router_port = free_port()
    metrics_port = free_port()
    prefill_url = f"http://127.0.0.1:{prefill_port}"
    decode_url = f"http://127.0.0.1:{decode_port}"
    router_url = f"http://127.0.0.1:{router_port}"

    log_dir = Path(tempfile.mkdtemp(prefix="nano_pd_router_"))
    prefill_log = log_dir / "prefill.log"
    decode_log = log_dir / "decode.log"
    router_log = log_dir / "router.log"

    prefill_proc = decode_proc = router_proc = None
    prefill_handle = decode_handle = router_handle = None
    client = None
    results = None
    ok = True
    try:
        prefill_handle = open(prefill_log, "w")
        decode_handle = open(decode_log, "w")
        router_handle = open(router_log, "w")
        print(f"model={path}")
        print(f"vllm_router={router_bin}")
        print(f"prefill_worker={prefill_url} device={prefill_device}")
        print(f"decode_worker={decode_url} device={decode_device}")
        print(f"router={router_url} logs={log_dir}")

        prefill_proc = subprocess.Popen(
            worker_command(
                path,
                "prefill",
                prefill_port,
                prefill_device,
                "prefill-1",
                model_args,
            ),
            cwd=str(REPO_ROOT),
            stdout=prefill_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        decode_proc = subprocess.Popen(
            worker_command(
                path,
                "decode",
                decode_port,
                decode_device,
                "decode-1",
                model_args,
            ),
            cwd=str(REPO_ROOT),
            stdout=decode_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0), trust_env=False)
        wait_http(client, f"{prefill_url}/health", prefill_proc, prefill_log, timeout, "prefill worker")
        wait_http(client, f"{decode_url}/health", decode_proc, decode_log, timeout, "decode worker")

        router_proc = subprocess.Popen(
            [
                router_bin,
                "--host",
                "127.0.0.1",
                "--port",
                str(router_port),
                "--vllm-pd-disaggregation",
                "--prefill",
                prefill_url,
                "none",
                "--decode",
                decode_url,
                "--policy",
                os.environ.get("NANO_VLLM_ROUTER_POLICY", "round_robin"),
                "--worker-startup-timeout-secs",
                str(int(timeout)),
                "--worker-startup-check-interval",
                "2",
                "--prometheus-host",
                "127.0.0.1",
                "--prometheus-port",
                str(metrics_port),
                "--log-level",
                os.environ.get("NANO_VLLM_ROUTER_LOG_LEVEL", "info"),
            ],
            cwd=str(REPO_ROOT),
            stdout=router_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_http(client, f"{router_url}/health", router_proc, router_log, timeout, "vllm-router")
        print("router is healthy")

        def send(prompt):
            response = client.post(
                f"{router_url}/v1/completions",
                json={
                    "model": path,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            return prompt, response

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            results = list(pool.map(send, prompts))

        for prompt, response in results:
            print(f"prompt={prompt!r} status={response.status_code}")
            print(f"response={response.text[:2000]}")
            if response.status_code != 200:
                ok = False
                continue
            body = response.json()
            if not body.get("pd_transfer"):
                print("ERROR: response did not use PD transfer", file=sys.stderr)
                ok = False
            if body.get("worker_role") != "decode":
                print("ERROR: response did not come from a decode worker", file=sys.stderr)
                ok = False
            choices = body.get("choices") or []
            if not choices or not choices[0].get("text"):
                print("ERROR: empty completion text", file=sys.stderr)
                ok = False
        if not ok:
            return 1

        chat_response = client.post(
            f"{router_url}/v1/chat/completions",
            json={
                "model": path,
                "messages": [{"role": "user", "content": "What is 1+1?"}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
        )
        print(f"chat_status={chat_response.status_code}")
        print(f"chat_response={chat_response.text[:2000]}")
        if chat_response.status_code != 200:
            print("ERROR: chat completion failed", file=sys.stderr)
            ok = False
        else:
            chat_body = chat_response.json()
            if not chat_body.get("pd_transfer"):
                print("ERROR: chat response did not use PD transfer", file=sys.stderr)
                ok = False
            choices = chat_body.get("choices") or []
            if not choices or not choices[0].get("message", {}).get("content"):
                print("ERROR: empty chat completion content", file=sys.stderr)
                ok = False
        if not ok:
            return 1

        stream_count = 0
        done = False
        with client.stream(
            "POST",
            f"{router_url}/v1/completions",
            json={
                "model": path,
                "prompt": prompts[0],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
        ) as stream:
            for line in stream.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    continue
                stream_count += 1
        print(f"stream_events={stream_count} stream_done={done}")
        if stream_count < 1 or not done:
            print("ERROR: streaming request did not complete", file=sys.stderr)
            return 1
    finally:
        if client is not None:
            client.close()
        stop_process(router_proc)
        stop_process(prefill_proc)
        stop_process(decode_proc)
        for handle in (prefill_handle, decode_handle, router_handle):
            if handle is not None:
                handle.close()
        if not ok:
            print(f"prefill_log:\n{read_log(prefill_log)}", file=sys.stderr)
            print(f"decode_log:\n{read_log(decode_log)}", file=sys.stderr)
            print(f"router_log:\n{read_log(router_log)}", file=sys.stderr)

    if results is None:
        return 1
    direct_texts = direct_generate(path, prompts, max_tokens, prefill_device, engine_kwargs)
    for (prompt, response), direct_text in zip(results, direct_texts):
        router_text = response.json()["choices"][0]["text"]
        match = router_text == direct_text
        print(f"compare prompt={prompt!r} match={match}")
        print(f"router_text={router_text!r}")
        print(f"direct_text={direct_text!r}")
        if not match:
            ok = False
    print(f"ALL_MATCH={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
