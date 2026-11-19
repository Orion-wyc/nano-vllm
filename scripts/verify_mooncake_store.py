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


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def find_master():
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


def wait_port(port, proc, log_path, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"mooncake_master exited with code {proc.returncode}\n{read_log(log_path)}"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"mooncake_master did not listen on {port}\n{read_log(log_path)}")


def read_log(path):
    try:
        return Path(path).read_text(errors="ignore")[-8000:]
    except OSError:
        return ""


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def main():
    try:
        from mooncake.store import MooncakeDistributedStore
    except Exception:
        traceback.print_exc()
        print("Mooncake Python package is not installed", file=sys.stderr)
        return 1

    master_bin = find_master()
    rpc_port = free_port()
    metrics_port = free_port()
    log_dir = Path(tempfile.mkdtemp(prefix="nano_mooncake_store_"))
    log_path = log_dir / "master.log"
    log_handle = open(log_path, "w")
    master_proc = None
    store = None
    try:
        master_proc = subprocess.Popen(
            [
                master_bin,
                f"--rpc_port={rpc_port}",
                f"--metrics_port={metrics_port}",
                "--enable_metric_reporting=false",
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_port(rpc_port, master_proc, log_path)

        store = MooncakeDistributedStore()
        status = store.setup(
            "127.0.0.1",
            "P2PHANDSHAKE",
            256 * 1024 * 1024,
            64 * 1024 * 1024,
            "tcp",
            "",
            f"127.0.0.1:{rpc_port}",
        )
        if status != 0:
            raise RuntimeError(f"MooncakeDistributedStore.setup failed: {status}")

        results = []
        for size_mb in (3, 20):
            value = os.urandom(size_mb * 1024 * 1024)
            key = f"nano:verify:{size_mb}m"
            start = time.perf_counter()
            put_status = store.put(key, value)
            put_sec = time.perf_counter() - start
            start = time.perf_counter()
            loaded = store.get(key)
            get_sec = time.perf_counter() - start
            results.append({
                "size_mb": size_mb,
                "put_status": put_status,
                "put_sec": round(put_sec, 4),
                "get_sec": round(get_sec, 4),
                "match": loaded == value,
            })

        keys = [f"nano:verify:batch:{index}" for index in range(4)]
        values = [os.urandom(3 * 1024 * 1024) for _ in keys]
        batch_put_status = store.put_batch(keys, values)
        batch_values = store.get_batch(keys)
        exists = store.batch_is_exist(keys + ["nano:verify:missing"])
        batch_match = batch_values == values

        print(f"master={master_bin}")
        print(f"master_rpc_port={rpc_port}")
        print(f"log={log_path}")
        print(f"objects={json.dumps(results)}")
        print(f"batch_put_status={batch_put_status}")
        print(f"batch_match={batch_match}")
        print(f"batch_exists={exists}")
        if not all(item["match"] and item["put_status"] == 0 for item in results):
            print("ERROR: object round trip failed", file=sys.stderr)
            return 1
        if batch_put_status != 0 or not batch_match or exists != [1, 1, 1, 1, 0]:
            print("ERROR: batch round trip failed", file=sys.stderr)
            return 1
        print("Mooncake store verification passed")
        return 0
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        stop_process(master_proc)
        log_handle.close()
        if master_proc is not None and master_proc.returncode not in (0, None, -15):
            print(f"mooncake_master log:\n{read_log(log_path)}", file=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
