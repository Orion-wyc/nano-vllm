import os
import sys
import traceback


def main():
    print("python", sys.version)
    print("ASCEND_HOME_PATH", os.environ.get("ASCEND_HOME_PATH"))
    try:
        import torch

        print("torch", torch.__version__)
        import torch_npu

        print("torch_npu", torch_npu.__version__)
        print("npu available", torch.npu.is_available())
        print("npu count", torch.npu.device_count())
        if torch.npu.is_available():
            print("device", torch.npu.get_device_name(0))
            a = torch.randn(1024, 1024, device="npu")
            b = a @ a
            torch.npu.synchronize()
            print("matmul sum", b.sum().item())
        import transformers

        print("transformers", transformers.__version__)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
