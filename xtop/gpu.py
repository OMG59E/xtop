"""GPU backends. Prefer NVML and fall back to an isolated nvidia-smi caller."""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import replace
from typing import Any

from .collect import GpuProc, GpuStat, _gpu_cmd, _gpu_user, _num


class NullGpuProvider:
    name = "none"
    metric_interval = 30.0
    process_interval = 30.0

    def metrics(self) -> tuple[list[GpuStat], str]:
        return [], ""

    def processes(self, gpus: list[GpuStat]) -> tuple[list[GpuStat], str]:
        return gpus, ""

    def close(self) -> None:
        pass


class NvidiaSmiProvider:
    name = "nvidia-smi"
    metric_interval = 2.0
    process_interval = 4.0

    def __init__(self, binary: str) -> None:
        self.binary = binary
        self.uuid_to_index: dict[str, int] = {}

    @staticmethod
    def _error(proc: subprocess.CompletedProcess[str]) -> str:
        lines = (proc.stderr or proc.stdout or "nvidia-smi failed").strip().splitlines()
        message = lines[0] if lines else "nvidia-smi failed"
        if "no devices were found" in message.lower():
            return ""
        return message

    def metrics(self) -> tuple[list[GpuStat], str]:
        query = [
            self.binary,
            "--query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(query, capture_output=True, text=True, timeout=2.0, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], str(exc)
        if proc.returncode != 0:
            return [], self._error(proc)

        gpus: list[GpuStat] = []
        uuid_to_index: dict[str, int] = {}
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            cols = [cell.strip() for cell in next(csv.reader([line]))]
            if len(cols) < 10:
                continue
            try:
                index = int(cols[0])
            except ValueError:
                continue
            uuid_to_index[cols[1]] = index
            gpus.append(
                GpuStat(
                    index=index,
                    name=cols[2],
                    util=_num(cols[3]),
                    mem_util=_num(cols[4]),
                    mem_used=_num(cols[5]),
                    mem_total=_num(cols[6]),
                    temp=_num(cols[7]),
                    power=_num(cols[8]),
                    power_limit=_num(cols[9]),
                    uuid=cols[1],
                )
            )
        self.uuid_to_index = uuid_to_index
        return sorted(gpus, key=lambda gpu: gpu.index), ""

    def processes(self, gpus: list[GpuStat]) -> tuple[list[GpuStat], str]:
        if not gpus:
            return [], ""
        query = [
            self.binary,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(query, capture_output=True, text=True, timeout=2.0, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return gpus, str(exc)
        if proc.returncode != 0:
            return gpus, self._error(proc)

        by_index = {gpu.index: replace(gpu, procs=[]) for gpu in gpus}
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            cols = [cell.strip() for cell in next(csv.reader([line]))]
            if len(cols) < 4:
                continue
            index = self.uuid_to_index.get(cols[0])
            if index is None or index not in by_index:
                continue
            try:
                pid = int(cols[1])
            except ValueError:
                continue
            by_index[index].procs.append(
                GpuProc(index, pid, _gpu_user(pid), _num(cols[3]), _gpu_cmd(pid, cols[2]))
            )
        return [by_index[gpu.index] for gpu in gpus], ""

    def close(self) -> None:
        pass


class NvmlProvider:
    name = "nvml"
    metric_interval = 1.0
    process_interval = 2.0

    def __init__(self, pynvml: Any) -> None:
        self.nvml = pynvml
        self.nvml.nvmlInit()
        self.handles: dict[int, Any] = {}

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode(errors="replace") if isinstance(value, bytes) else str(value)

    def _value(self, call: Any, scale: float = 1.0) -> float | None:
        try:
            return float(call()) / scale
        except (self.nvml.NVMLError, AttributeError):
            return None

    def metrics(self) -> tuple[list[GpuStat], str]:
        try:
            count = self.nvml.nvmlDeviceGetCount()
        except self.nvml.NVMLError as exc:
            return [], str(exc)

        rows: list[GpuStat] = []
        handles: dict[int, Any] = {}
        device_errors: list[str] = []
        for index in range(count):
            try:
                handle = self.nvml.nvmlDeviceGetHandleByIndex(index)
                name = self._text(self.nvml.nvmlDeviceGetName(handle))
            except self.nvml.NVMLError as exc:
                device_errors.append(f"GPU{index}: {exc}")
                continue
            handles[index] = handle
            try:
                uuid = self._text(self.nvml.nvmlDeviceGetUUID(handle))
            except self.nvml.NVMLError:
                uuid = ""
            try:
                util = self.nvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util, mem_util = float(util.gpu), float(util.memory)
            except self.nvml.NVMLError:
                gpu_util = mem_util = None
            try:
                memory = self.nvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = float(memory.used) / 1024 / 1024
                mem_total = float(memory.total) / 1024 / 1024
            except self.nvml.NVMLError:
                mem_used = mem_total = None
            rows.append(
                GpuStat(
                    index=index,
                    name=name,
                    util=gpu_util,
                    mem_util=mem_util,
                    mem_used=mem_used,
                    mem_total=mem_total,
                    temp=self._value(
                        lambda: self.nvml.nvmlDeviceGetTemperature(
                            handle, self.nvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                    power=self._value(lambda: self.nvml.nvmlDeviceGetPowerUsage(handle), 1000.0),
                    power_limit=self._value(
                        lambda: self.nvml.nvmlDeviceGetEnforcedPowerLimit(handle), 1000.0
                    ),
                    uuid=uuid,
                )
            )
        self.handles = handles
        return rows, "; ".join(device_errors)

    def processes(self, gpus: list[GpuStat]) -> tuple[list[GpuStat], str]:
        rows: list[GpuStat] = []
        for gpu in gpus:
            current = replace(gpu, procs=[])
            handle = self.handles.get(gpu.index)
            if handle is None:
                rows.append(current)
                continue
            try:
                processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except self.nvml.NVMLError as exc:
                unsupported = getattr(self.nvml, "NVMLError_NotSupported", ())
                if unsupported and isinstance(exc, unsupported):
                    processes = []
                else:
                    return gpus, str(exc)
            for proc in processes:
                used = getattr(proc, "usedGpuMemory", None)
                unavailable = getattr(self.nvml, "NVML_VALUE_NOT_AVAILABLE", None)
                mem_mib = None if used is None or used == unavailable else float(used) / 1024 / 1024
                current.procs.append(
                    GpuProc(gpu.index, proc.pid, _gpu_user(proc.pid), mem_mib, _gpu_cmd(proc.pid, ""))
                )
            rows.append(current)
        return rows, ""

    def close(self) -> None:
        try:
            self.nvml.nvmlShutdown()
        except self.nvml.NVMLError:
            pass


def make_gpu_provider() -> NullGpuProvider | NvidiaSmiProvider | NvmlProvider:
    try:
        import pynvml  # type: ignore[import-not-found]

        return NvmlProvider(pynvml)
    except Exception as exc:
        # pynvml uses an exception hierarchy that is unavailable when import
        # itself fails, so backend discovery intentionally catches all errors.
        _ = exc
    binary = shutil.which("nvidia-smi")
    return NvidiaSmiProvider(binary) if binary else NullGpuProvider()
