"""Houmo IPU metrics through the postmo_smi C API in libhal_xh2a."""

from __future__ import annotations

import ctypes
import math
from ctypes import (
    POINTER,
    c_char,
    c_float,
    c_int,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)

from .collect import IpuStat

HAL_LIBRARY = "/usr/local/houmo-sdk/hal/lib/libhal_xh2a.so"
STRING_LEN = 256
STRUCT_VERSION = 1
MAX_DEVICES = 64
MIB = 1024 * 1024


class _Metrics(ctypes.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("size", c_uint32),
        ("temperature_c", c_float),
        ("power_w", c_float),
        ("average_power_w", c_float),
        ("ipu_util_percent", c_float),
        ("mem_used_bytes", c_uint64),
        ("mem_total_bytes", c_uint64),
        ("governor", c_char * STRING_LEN),
        ("pcie_bandwidth", c_char * STRING_LEN),
        ("reserved", c_uint8 * 32),
    ]


class IpuProvider:
    name = "postmo-smi"
    interval = 1.0

    def __init__(self, library: str = HAL_LIBRARY) -> None:
        self.ctx = c_void_p()
        self.lib = ctypes.CDLL(library)
        self.lib.postmo_smi_init.argtypes = [c_uint32, POINTER(c_void_p)]
        self.lib.postmo_smi_init.restype = c_int
        self.lib.postmo_smi_shutdown.argtypes = [c_void_p]
        self.lib.postmo_smi_shutdown.restype = c_int
        self.lib.postmo_smi_device_get_count.argtypes = [c_void_p, POINTER(c_uint32)]
        self.lib.postmo_smi_device_get_count.restype = c_int
        self.lib.postmo_smi_device_get_handle_by_index.argtypes = [
            c_void_p, c_uint32, POINTER(c_void_p),
        ]
        self.lib.postmo_smi_device_get_handle_by_index.restype = c_int
        self.lib.postmo_smi_device_get_metrics.argtypes = [c_void_p, POINTER(_Metrics)]
        self.lib.postmo_smi_device_get_metrics.restype = c_int
        self.lib.postmo_smi_device_get_power.argtypes = [c_void_p, POINTER(c_float)]
        self.lib.postmo_smi_device_get_power.restype = c_int

        status = self.lib.postmo_smi_init(0, ctypes.byref(self.ctx))
        if status != 0:
            raise OSError(f"postmo_smi_init failed: {status}")
        if not self.ctx:
            raise OSError("postmo_smi_init returned an empty context")

    @staticmethod
    def _finite(value: float, minimum: float | None = None, maximum: float | None = None) -> float | None:
        value = float(value)
        if not math.isfinite(value):
            return None
        if minimum is not None and value < minimum:
            return None
        if maximum is not None and value > maximum:
            return None
        return value

    @staticmethod
    def _error(label: str, error: Exception) -> str:
        detail = str(error).strip() or error.__class__.__name__
        return f"{label}: {detail}"

    def metrics(self) -> tuple[list[IpuStat], str]:
        if not self.ctx:
            return [], "postmo_smi is not initialized"

        count = c_uint32()
        try:
            status = self.lib.postmo_smi_device_get_count(self.ctx, ctypes.byref(count))
        except Exception as error:  # ctypes turns loader/ABI failures into several exception types.
            return [], self._error("postmo_smi device count failed", error)
        if status != 0:
            return [], f"postmo_smi device count failed: {status}"
        if count.value > MAX_DEVICES:
            return [], f"postmo_smi returned invalid device count: {count.value}"

        rows: list[IpuStat] = []
        errors: list[str] = []
        for index in range(count.value):
            device = c_void_p()
            try:
                status = self.lib.postmo_smi_device_get_handle_by_index(
                    self.ctx, index, ctypes.byref(device)
                )
            except Exception as error:
                errors.append(self._error(f"IPU{index} handle", error))
                continue
            if status != 0:
                errors.append(f"IPU{index}: handle {status}")
                continue
            if not device:
                errors.append(f"IPU{index}: empty handle")
                continue
            sample = _Metrics()
            sample.version = STRUCT_VERSION
            sample.size = ctypes.sizeof(_Metrics)
            # The API uses NaN for optional metrics on devices that are not ready.
            sample.temperature_c = math.nan
            sample.power_w = math.nan
            sample.average_power_w = math.nan
            sample.ipu_util_percent = math.nan
            try:
                status = self.lib.postmo_smi_device_get_metrics(device, ctypes.byref(sample))
            except Exception as error:
                errors.append(self._error(f"IPU{index} metrics", error))
                continue
            if status != 0:
                errors.append(f"IPU{index}: metrics {status}")
                continue

            # Memory is reported in bytes by postmo_smi; the UI displays MiB.
            mem_used = int(sample.mem_used_bytes // MIB)
            mem_total = int(sample.mem_total_bytes // MIB)

            power = c_float(math.nan)
            try:
                power_status = self.lib.postmo_smi_device_get_power(
                    device, ctypes.byref(power)
                )
            except Exception:
                power_status = -1
            power_value = self._finite(
                power.value if power_status == 0 else sample.power_w,
                minimum=0.0,
            )
            util = self._finite(sample.ipu_util_percent, 0.0, 100.0)
            if util is None and mem_total == 0 and power_value is None:
                # Per the API, probe-failed/HAL-not-ready devices return SUCCESS
                # with NaN and empty fields instead of an error status.
                errors.append(f"IPU{index}: telemetry unavailable")
                continue
            rows.append(IpuStat(index, util, mem_used, mem_total, power_value))
        return rows, "; ".join(errors)

    def close(self) -> None:
        ctx = self.ctx
        self.ctx = c_void_p()
        if not ctx:
            return
        try:
            self.lib.postmo_smi_shutdown(ctx)
        except Exception:
            # Shutdown runs during terminal cleanup and must not mask the real error.
            pass


def make_ipu_provider() -> IpuProvider | None:
    try:
        return IpuProvider()
    except Exception:
        # Missing SDK/library/device is a supported configuration.
        return None
