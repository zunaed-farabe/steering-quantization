"""Small environment diagnostic. Does not download weights or expose credentials."""

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def diagnostic():
    report = {"python": platform.python_version(), "platform": platform.platform(), "packages": {}, "gpus": [], "problems": []}
    for package in ("steerquant", "torch", "transformers", "accelerate", "bitsandbytes", "numpy"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
            report["problems"].append(f"Missing package: {package}")
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["pytorch_cuda_runtime"] = torch.version.cuda
        if not torch.cuda.is_available():
            report["problems"].append("No CUDA GPU available to this Python process")
        else:
            report["pytorch_arch_list"] = torch.cuda.get_arch_list()
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    free, total = torch.cuda.mem_get_info(index)
                    major, minor = torch.cuda.get_device_capability(index)
                    report["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                           "compute_capability": f"{major}.{minor}",
                                           "total_gib": total / 1024**3, "free_gib": free / 1024**3,
                                           "bf16_supported": bool(torch.cuda.is_bf16_supported())})
            gpu = report["gpus"][0]
            if not gpu["bf16_supported"]:
                report["problems"].append("Visible cuda:0 does not support the requested BF16 reference")
            arch = gpu["compute_capability"].replace(".", "")
            if not {f"sm_{arch}", f"compute_{arch}"} & set(report["pytorch_arch_list"]):
                # e.g. Blackwell (sm_120/sm_100) needs a PyTorch build for CUDA 12.8 or newer.
                report["problems"].append(f"This PyTorch build has no kernels for sm_{arch}; install a build for a newer CUDA")
            else:
                kernel_checks(torch, report)
    except Exception as exc:
        report["problems"].append(f"PyTorch diagnostic failed: {type(exc).__name__}: {exc}")
    report["ready_for_gpu_check"] = not report["problems"]
    return report


def kernel_checks(torch, report):
    """Run tiny real BF16, LLM.int8() and NF4 layers so kernel problems show up in seconds."""
    checks = report["kernel_checks"] = {}
    try:
        x = torch.randn(4, 64, device="cuda:0", dtype=torch.bfloat16)
        checks["bf16_matmul"] = bool(torch.isfinite(x @ x.T).all())
    except Exception as exc:
        checks["bf16_matmul"] = f"{type(exc).__name__}: {exc}"
    if report["packages"].get("bitsandbytes"):
        try:
            import bitsandbytes as bnb
            layer = bnb.nn.Linear8bitLt(64, 64, bias=False, has_fp16_weights=False, threshold=6.0).to("cuda:0")
            checks["int8_linear"] = bool(torch.isfinite(layer(torch.randn(4, 64, device="cuda:0", dtype=torch.float16))).all())
        except Exception as exc:
            checks["int8_linear"] = f"{type(exc).__name__}: {exc}"
        try:
            import bitsandbytes as bnb
            layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4").to("cuda:0")
            checks["nf4_linear"] = bool(torch.isfinite(layer(torch.randn(4, 64, device="cuda:0", dtype=torch.bfloat16))).all())
        except Exception as exc:
            checks["nf4_linear"] = f"{type(exc).__name__}: {exc}"
    for name, result in checks.items():
        if result is not True:
            report["problems"].append(f"Kernel check {name} failed: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="logs/gpu_environment.json")
    args = parser.parse_args()
    report = diagnostic()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ready_for_gpu_check"] else 1


if __name__ == "__main__":
    sys.exit(main())
