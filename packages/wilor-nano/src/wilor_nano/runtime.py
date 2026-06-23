import torch


def get_torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_torch_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "cuda" else torch.float32


def get_rtmpose_device() -> str:
    try:
        import onnxruntime as ort
    except ImportError:
        return "cuda" if torch.cuda.is_available() else "cpu"

    providers = set(ort.get_available_providers())
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"
