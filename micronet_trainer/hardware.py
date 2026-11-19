"""Hardware: Übersicht vor dem Start, GPU-Auslastung, Standby-Sperre (Windows)."""

import contextlib
import platform
import shutil
import sys
from pathlib import Path

import yaml

GB = 1024 ** 3


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unbekannt"


def resolve_device(setting: str) -> str:
    """'cuda' oder 'cpu' (compat.predict erkennt die GPU am Namen 'cuda')."""
    import torch
    if setting == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        if setting.startswith("cuda:"):
            torch.cuda.set_device(int(setting.split(":")[1]))
        return "cuda"
    return "cpu"


def auto_workers(setting) -> int:
    if setting != "auto":
        return int(setting)
    import psutil
    cores = psutil.cpu_count(logical=False) or 2
    return max(1, min(6, cores - 1))


def collect(device: str, output: Path, workers: int, batch_size: int | None) -> dict:
    import psutil
    import torch
    memory = psutil.virtual_memory()
    output.mkdir(parents=True, exist_ok=True)
    info = {
        "system": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "cpu": f"{cpu_name()}, {psutil.cpu_count(logical=False)} Kerne / {psutil.cpu_count()} Threads",
        "ram_gb": round(memory.total / GB, 1),
        "ram_free_gb": round(memory.available / GB, 1),
        "torch": str(torch.__version__),        # TorchVersion-Objekt -> Text (YAML kann es sonst nicht)
        "device": device,
        "disk_free_gb": round(shutil.disk_usage(output).free / GB, 1),
        "dataloader_workers": workers,
        "batch_size": batch_size,
    }
    try:
        import segmentation_models_pytorch as smp
        info["smp"] = str(smp.__version__)
    except ImportError:
        info["smp"] = "fehlt"
    if device == "cuda":
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        info.update({
            "gpu": props.name,
            "gpu_memory_gb": round(total / GB, 1),
            "gpu_memory_free_gb": round(free / GB, 1),
            "gpu_compute": f"{props.major}.{props.minor}",
            "cuda": str(torch.version.cuda),
            "cudnn": str(torch.backends.cudnn.version()),
            "amp_float16": True,
        })
    return info


def describe(info: dict) -> str:
    lines = ["=== Hardware ===",
             f"System      {info['system']}, Python {info['python']}",
             f"CPU         {info['cpu']}",
             f"RAM         {info['ram_gb']} GB gesamt, {info['ram_free_gb']} GB frei"]
    if info["device"] == "cuda":
        lines += [f"GPU         {info['gpu']}, {info['gpu_memory_gb']} GB VRAM ({info['gpu_memory_free_gb']} GB frei), "
                  f"Compute {info['gpu_compute']}   <- wird benutzt",
                  f"CUDA        {info['cuda']}, cuDNN {info['cudnn']}, AMP float16: ja"]
    else:
        lines += ["GPU         KEINE CUDA-GPU – Training auf der CPU (sehr langsam!)"]
    lines += [f"PyTorch     {info['torch']}, smp {info['smp']}",
              f"Datenträger {info['disk_free_gb']} GB frei im Ausgabeordner",
              f"DataLoader  {info['dataloader_workers']} Worker, Batch {info['batch_size']} Patches"]
    return "\n".join(lines)


def plain(value):
    """Nur einfache Typen für YAML: Zahlen-/Text-Unterklassen (numpy, TorchVersion, Path …) umwandeln."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):              # numpy-Skalar
        return plain(value.item())
    return str(value)


def save(info: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(plain(info), sort_keys=False, allow_unicode=True), encoding="utf-8")


class GpuMonitor:
    """GPU-Auslastung in % und belegter Speicher in GB (nvidia-ml-py); ohne Treiber: None."""

    def __init__(self, device: str):
        self.nvml = None
        if device != "cuda":
            return
        try:
            import pynvml
            import torch
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            self.nvml = pynvml
        except Exception:       # noqa: BLE001 – Auslastung ist nur Zusatzinformation
            self.nvml = None

    def sample(self) -> tuple[float | None, float | None]:
        if self.nvml is None:
            return None, None
        try:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
            used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used / GB
            return float(util), round(used, 1)
        except Exception:       # noqa: BLE001
            return None, None


@contextlib.contextmanager
def prevent_sleep(enabled: bool):
    """Windows: kein Standby, solange der Lauf läuft (Bildschirm darf aus)."""
    active = False
    if enabled and sys.platform == "win32":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        active = bool(ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required))
    try:
        yield active
    finally:
        if active:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
