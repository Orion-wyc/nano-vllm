import functools

import torch


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:
        return False


def device_type() -> str:
    if _npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_npu() -> bool:
    return device_type() == "npu"


def is_cuda() -> bool:
    return device_type() == "cuda"


def device(index: int = 0) -> torch.device:
    kind = device_type()
    if kind == "cpu":
        return torch.device("cpu")
    return torch.device(f"{kind}:{index}")


def set_device(index: int) -> None:
    kind = device_type()
    if kind == "npu":
        torch.npu.set_device(index)
    elif kind == "cuda":
        torch.cuda.set_device(index)


def synchronize(index: int | None = None) -> None:
    kind = device_type()
    if kind == "npu":
        torch.npu.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()


def empty_cache() -> None:
    kind = device_type()
    if kind == "npu":
        torch.npu.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()


def reset_peak_memory_stats() -> None:
    kind = device_type()
    if kind == "npu":
        torch.npu.reset_peak_memory_stats()
    elif kind == "cuda":
        torch.cuda.reset_peak_memory_stats()


def mem_get_info() -> tuple[int, int]:
    kind = device_type()
    if kind == "npu":
        return torch.npu.mem_get_info()
    if kind == "cuda":
        return torch.cuda.mem_get_info()
    return (0, 0)


def memory_stats() -> dict:
    kind = device_type()
    if kind == "npu":
        return torch.npu.memory_stats()
    if kind == "cuda":
        return torch.cuda.memory_stats()
    return {}


def memory_allocated() -> int:
    kind = device_type()
    if kind == "npu":
        return torch.npu.memory_allocated()
    if kind == "cuda":
        return torch.cuda.memory_allocated()
    return 0


def dist_backend() -> str:
    kind = device_type()
    if kind == "npu":
        return "hccl"
    if kind == "cuda":
        return "nccl"
    return "gloo"


def to_device(tensor: torch.Tensor, index: int = 0, non_blocking: bool = True) -> torch.Tensor:
    kind = device_type()
    if kind == "cpu":
        return tensor
    return tensor.to(device(index), non_blocking=non_blocking)


def tensor_from_list(values, dtype: torch.dtype, index: int = 0) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=dtype)
    return to_device(tensor, index)


def maybe_compile(fn):
    if not torch.cuda.is_available():
        return fn
    return torch.compile(fn)


def lru_cache_no_compile(fn):
    return functools.lru_cache(1)(fn)
