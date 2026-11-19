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
    candidate = REPO_ROOT.parent / "vllm-router" / "target" / "release" / "vllm-router"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    raise FileNotFoundError("vllm-router not found; set VLLM_ROUTER_BIN")


def find_master_bin():
    env_path = os.environ.get("MOONCAKE_MASTER_BIN")
    if env_path:
        return env_path
    found = shutil.which("mooncake_master")
    if found:
        return found
    candidate = Path(sys.executable).parent / "mooncake_master"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    raise FileNotFoundError("mooncake_master not found; run scripts/setup_mooncake_env.sh")


def read_log(path):
    try:
        return Path(path).read_text(errors="ignore")[-8000:]
    except OSError:
        return ""


def wait_http(client, url, proc, log_path, timeout, name):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
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


def wait_port(port, proc, log_path, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"mooncake_master exited with code {proc.returncode}\n{read_log(log_path)}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"mooncake_master did not listen on {port}\n{read_log(log_path)}")


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def worker_command(path, role, port, device, worker_id, master_port, bootstrap_port, model_args):
    command = [
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
        "--enforce-eager",
        "--kv-store",
        "mooncake",
        "--mooncake-master",
        f"127.0.0.1:{master_port}",
        "--mooncake-state-snapshots",
        os.environ.get("NANO_VLLM_MOONCAKE_STATE_SNAPSHOTS", "block"),
        *model_args,
    ]
    if role == "prefill":
        command += [
            "--advertise-url",
            f"http://127.0.0.1:{port}",
            "--mooncake-bootstrap-port",
            str(bootstrap_port),
        ]
    return command


def build_prompts():
    prefix = "The quick brown fox jumps over the lazy dog. " * 40
    return [
        prefix + " Answer A.",
        prefix + " Answer B.",
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
    master_bin = find_master_bin()
    prefill_device = int(os.environ.get("NANO_VLLM_PREFILL_DEVICE", "0"))
    decode_device = int(os.environ.get("NANO_VLLM_DECODE_DEVICE", "1"))
    max_tokens = int(os.environ.get("NANO_VLLM_MAX_TOKENS", "8"))
    timeout = float(os.environ.get("NANO_VLLM_MOONCAKE_TIMEOUT", "900"))
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
    prompts = build_prompts()

    master_port = free_port()
    prefill_port = free_port()
    decode_port = free_port()
    router_port = free_port()
    bootstrap_port = free_port()
    metrics_port = free_port()
    prefill_url = f"http://127.0.0.1:{prefill_port}"
    decode_url = f"http://127.0.0.1:{decode_port}"
    router_url = f"http://127.0.0.1:{router_port}"

    log_dir = Path(tempfile.mkdtemp(prefix="nano_mooncake_pd_"))
    master_log = log_dir / "mooncake_master.log"
    prefill_log = log_dir / "prefill.log"
    decode_log = log_dir / "decode.log"
    router_log = log_dir / "router.log"

    master_proc = prefill_proc = decode_proc = router_proc = None
    handles = []
    client = None
    responses = []
    ok = True
    try:
        print(f"model={path}")
        print(f"vllm_router={router_bin}")
        print(f"mooncake_master={master_bin} port={master_port}")
        print(f"prefill_worker={prefill_url} device={prefill_device} bootstrap={bootstrap_port}")
        print(f"decode_worker={decode_url} device={decode_device}")
        print(f"router={router_url} logs={log_dir}")

        master_handle = open(master_log, "w")
        handles.append(master_handle)
        master_proc = subprocess.Popen(
            [
                master_bin,
                f"--rpc_port={master_port}",
                f"--metrics_port={metrics_port}",
                "--enable_metric_reporting=false",
            ],
            stdout=master_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_port(master_port, master_proc, master_log)

        prefill_handle = open(prefill_log, "w")
        handles.append(prefill_handle)
        prefill_proc = subprocess.Popen(
            worker_command(
                path,
                "prefill",
                prefill_port,
                prefill_device,
                "prefill-1",
                master_port,
                bootstrap_port,
                model_args,
            ),
            cwd=str(REPO_ROOT),
            stdout=prefill_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        decode_handle = open(decode_log, "w")
        handles.append(decode_handle)
        decode_proc = subprocess.Popen(
            worker_command(
                path,
                "decode",
                decode_port,
                decode_device,
                "decode-1",
                master_port,
                bootstrap_port,
                model_args,
            ),
            cwd=str(REPO_ROOT),
            stdout=decode_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client = httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0), trust_env=False)
        wait_http(client, f"{prefill_url}/health", prefill_proc, prefill_log, timeout, "prefill worker")
        wait_http(client, f"{decode_url}/health", decode_proc, decode_log, timeout, "decode worker")
        wait_http(
            client,
            f"http://127.0.0.1:{bootstrap_port}/health",
            prefill_proc,
            prefill_log,
            timeout,
            "mooncake bootstrap",
        )

        router_handle = open(router_log, "w")
        handles.append(router_handle)
        router_proc = subprocess.Popen(
            [
                router_bin,
                "--host",
                "127.0.0.1",
                "--port",
                str(router_port),
                "--vllm-pd-disaggregation",
                "--kv-connector",
                "mooncake",
                "--prefill",
                prefill_url,
                str(bootstrap_port),
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
                str(metrics_port + 1),
                "--log-level",
                "info",
            ],
            cwd=str(REPO_ROOT),
            stdout=router_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_http(client, f"{router_url}/health", router_proc, router_log, timeout, "vllm-router")
        print("router is healthy")

        for index, prompt in enumerate(prompts):
            response = client.post(
                f"{router_url}/v1/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": False,
                },
            )
            print(f"request={index} status={response.status_code}")
            print(f"request={index} response={response.text[:2000]}")
            if response.status_code != 200:
                ok = False
                continue
            body = response.json()
            responses.append(body)
            if body.get("kv_store") != "mooncake":
                print("ERROR: response did not report Mooncake KV store", file=sys.stderr)
                ok = False
            if body.get("pd_transfer") is not True:
                print("ERROR: response did not report PD transfer", file=sys.stderr)
                ok = False
            if body.get("worker_role") != "decode":
                print("ERROR: response did not come from decode worker", file=sys.stderr)
                ok = False

        if len(responses) < 2:
            print("ERROR: not all requests returned", file=sys.stderr)
            return 1
        second_hit = int(responses[1].get("prefix_hit_blocks") or 0)
        print(f"second_prefix_hit_blocks={second_hit}")
        if second_hit < 1:
            print("ERROR: second request did not hit the Mooncake prefix", file=sys.stderr)
            ok = False

        prefill_log_text = read_log(prefill_log)
        http_transfer_count = prefill_log_text.count("/internal/pd/transfer")
        print(f"prefill_http_transfer_requests={http_transfer_count}")
        if http_transfer_count != 0:
            print("ERROR: prefill log still shows HTTP transfer requests", file=sys.stderr)
            ok = False
        if not ok:
            return 1
    finally:
        if client is not None:
            client.close()
        stop_process(router_proc)
        stop_process(prefill_proc)
        stop_process(decode_proc)
        stop_process(master_proc)
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass
        if not ok:
            print(f"prefill_log:\n{read_log(prefill_log)}", file=sys.stderr)
            print(f"decode_log:\n{read_log(decode_log)}", file=sys.stderr)
            print(f"router_log:\n{read_log(router_log)}", file=sys.stderr)
            print(f"master_log:\n{read_log(master_log)}", file=sys.stderr)

    direct_texts = direct_generate(path, prompts, max_tokens, prefill_device, engine_kwargs)
    for index, (body, direct_text) in enumerate(zip(responses, direct_texts)):
        router_text = body["choices"][0]["text"]
        match = router_text == direct_text
        print(f"compare request={index} match={match}")
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
