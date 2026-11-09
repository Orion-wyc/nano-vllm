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


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def worker_command(path, port, device, worker_id, model_args):
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
        "--worker-id",
        worker_id,
        "--enforce-eager",
        *model_args,
    ]


def start_worker(path, port, device, worker_id, model_args, log_path):
    handle = open(log_path, "w")
    proc = subprocess.Popen(
        worker_command(path, port, device, worker_id, model_args),
        cwd=str(REPO_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, handle


def start_router(router_bin, router_port, metrics_port, worker_urls, timeout, log_path):
    handle = open(log_path, "w")
    proc = subprocess.Popen(
        [
            router_bin,
            "--host",
            "127.0.0.1",
            "--port",
            str(router_port),
            "--worker-urls",
            *worker_urls,
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
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, handle


def send(client, router_url, prompt, max_tokens):
    response = client.post(
        f"{router_url}/v1/completions",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
    )
    return response


def check_response(response, expected_worker=None):
    if response.status_code != 200:
        raise RuntimeError(f"request failed: {response.status_code} {response.text[:1000]}")
    body = response.json()
    if body.get("worker_role") != "regular":
        raise RuntimeError(f"unexpected worker_role: {body.get('worker_role')}")
    if body.get("pd_transfer") is not False:
        raise RuntimeError(f"regular worker should not report PD transfer: {body.get('pd_transfer')}")
    worker_id = body.get("worker_id")
    if worker_id is None:
        raise RuntimeError("response did not include worker_id")
    if expected_worker is not None and worker_id != expected_worker:
        raise RuntimeError(f"expected worker {expected_worker}, got {worker_id}")
    choices = body.get("choices") or []
    if not choices or not choices[0].get("text"):
        raise RuntimeError("empty completion text")
    return worker_id


def run_scenario_single(client, router_bin, path, worker, model_args, timeout, log_dir):
    worker_port = free_port()
    router_port = free_port()
    metrics_port = free_port()
    worker_url = f"http://127.0.0.1:{worker_port}"
    router_url = f"http://127.0.0.1:{router_port}"
    worker_log = log_dir / "regular-1.log"
    router_log = log_dir / "router-single.log"
    worker_proc = worker_handle = router_proc = router_handle = None
    try:
        worker_proc, worker_handle = start_worker(
            path, worker_port, worker, "regular-1", model_args, worker_log
        )
        wait_http(client, f"{worker_url}/health", worker_proc, worker_log, timeout, "regular worker")
        router_proc, router_handle = start_router(
            router_bin, router_port, metrics_port, [worker_url], timeout, router_log
        )
        wait_http(client, f"{router_url}/health", router_proc, router_log, timeout, "vllm-router")
        for index in range(4):
            response = send(client, router_url, f"What is 1+1? request {index}", 4)
            worker_id = check_response(response, expected_worker="regular-1")
            print(f"S3 request={index} worker_id={worker_id} status=200")
    finally:
        stop_process(router_proc)
        stop_process(worker_proc)
        if router_handle is not None:
            router_handle.close()
        if worker_handle is not None:
            worker_handle.close()
        if not worker_log.exists() or not router_log.exists():
            return
        if worker_proc is not None and worker_proc.returncode not in (0, None, -15):
            print(f"S3 worker log:\n{read_log(worker_log)}", file=sys.stderr)
        if router_proc is not None and router_proc.returncode not in (0, None, -15):
            print(f"S3 router log:\n{read_log(router_log)}", file=sys.stderr)


def run_scenario_dp2(client, router_bin, path, devices, model_args, timeout, log_dir):
    ports = [free_port(), free_port()]
    router_port = free_port()
    metrics_port = free_port()
    worker_urls = [f"http://127.0.0.1:{port}" for port in ports]
    router_url = f"http://127.0.0.1:{router_port}"
    worker_logs = [log_dir / "regular-1.log", log_dir / "regular-2.log"]
    router_log = log_dir / "router-dp2.log"
    workers = []
    router_proc = router_handle = None
    try:
        for index, (port, device, log_path) in enumerate(zip(ports, devices, worker_logs)):
            worker_id = f"regular-{index + 1}"
            workers.append(start_worker(path, port, device, worker_id, model_args, log_path))
        for index, (proc, handle) in enumerate(workers):
            wait_http(
                client,
                f"{worker_urls[index]}/health",
                proc,
                worker_logs[index],
                timeout,
                f"regular worker {index + 1}",
            )
        router_proc, router_handle = start_router(
            router_bin, router_port, metrics_port, worker_urls, timeout, router_log
        )
        wait_http(client, f"{router_url}/health", router_proc, router_log, timeout, "vllm-router")
        counts = {}
        for index in range(8):
            response = send(client, router_url, f"What is 1+1? dp2 request {index}", 4)
            worker_id = check_response(response)
            counts[worker_id] = counts.get(worker_id, 0) + 1
            print(f"S4 request={index} worker_id={worker_id} status=200")
        print(f"S4 distribution={counts}")
        missing = {"regular-1", "regular-2"} - set(counts)
        if missing:
            raise RuntimeError(f"DP=2 did not hit workers: {sorted(missing)}")
        if max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError(f"round_robin distribution is unbalanced: {counts}")
    finally:
        stop_process(router_proc)
        for proc, handle in workers:
            stop_process(proc)
            handle.close()
        if router_handle is not None:
            router_handle.close()
        if router_proc is not None and router_proc.returncode not in (0, None, -15):
            print(f"S4 router log:\n{read_log(router_log)}", file=sys.stderr)
        for log_path, (proc, _) in zip(worker_logs, workers):
            if proc is not None and proc.returncode not in (0, None, -15):
                print(f"S4 worker log {log_path}:\n{read_log(log_path)}", file=sys.stderr)


def main():
    path = resolve_model_path()
    router_bin = find_router_bin()
    device_0 = int(os.environ.get("NANO_VLLM_REGULAR_DEVICE_0", "0"))
    device_1 = int(os.environ.get("NANO_VLLM_REGULAR_DEVICE_1", "1"))
    max_tokens = int(os.environ.get("NANO_VLLM_MAX_TOKENS", "4"))
    timeout = float(os.environ.get("NANO_VLLM_ROUTER_TIMEOUT", "600"))
    gpu_util = os.environ.get("NANO_VLLM_GPU_UTIL", "0.3")
    model_args = [
        "--max-model-len",
        os.environ.get("NANO_VLLM_MAX_MODEL_LEN", "512"),
        "--max-num-seqs",
        os.environ.get("NANO_VLLM_MAX_NUM_SEQS", "8"),
        "--max-num-batched-tokens",
        os.environ.get("NANO_VLLM_MAX_NUM_BATCHED_TOKENS", "512"),
        "--gpu-memory-utilization",
        gpu_util,
    ]
    log_dir = Path(tempfile.mkdtemp(prefix="nano_router_regular_"))
    print(f"model={path}")
    print(f"vllm_router={router_bin}")
    print(f"logs={log_dir}")
    client = httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0), trust_env=False)
    try:
        print("=== S3: single regular worker + vllm-router ===")
        run_scenario_single(client, router_bin, path, device_0, model_args, timeout, log_dir)
        print("=== S4: regular DP=2 + vllm-router ===")
        run_scenario_dp2(client, router_bin, path, [device_0, device_1], model_args, timeout, log_dir)
        print("ALL_OK")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
