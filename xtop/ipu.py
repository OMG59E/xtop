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


class _MemInfo(ctypes.Structure):
    _fields_ = [
        ("mem_total", c_uint32),
        ("mem_used", c_uint32),
        ("mem_avail", c_uint32),
    ]


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
        self.lib.hm_sys_get_mem_info.argtypes = [c_int, POINTER(_MemInfo)]
        self.lib.hm_sys_get_mem_info.restype = c_int
        self.lib.hm_sys_get_board_power.argtypes = [c_int, POINTER(c_float)]
        self.lib.hm_sys_get_board_power.restype = c_int

        ctx = c_void_p()
        status = self.lib.postmo_smi_init(0, ctypes.byref(ctx))
        if status != 0:
            raise OSError(f"postmo_smi_init failed: {status}")
        self.ctx = ctx

    def metrics(self) -> tuple[list[IpuStat], str]:
        count = c_uint32()
        status = self.lib.postmo_smi_device_get_count(self.ctx, ctypes.byref(count))
        if status != 0:
            return [], f"postmo_smi device count failed: {status}"

        rows: list[IpuStat] = []
        errors: list[str] = []
        for index in range(count.value):
            device = c_void_p()
            status = self.lib.postmo_smi_device_get_handle_by_index(
                self.ctx, index, ctypes.byref(device)
            )
            if status != 0:
                errors.append(f"IPU{index}: handle {status}")
                continue
            sample = _Metrics()
            sample.version = STRUCT_VERSION
            sample.size = ctypes.sizeof(_Metrics)
            status = self.lib.postmo_smi_device_get_metrics(device, ctypes.byref(sample))
            if status != 0:
                errors.append(f"IPU{index}: metrics {status}")
                continue
            memory = _MemInfo()
            if self.lib.hm_sys_get_mem_info(index, ctypes.byref(memory)) != 0:
                errors.append(f"IPU{index}: memory")
                continue
            power = c_float()
            power_status = self.lib.hm_sys_get_board_power(index, ctypes.byref(power))
            util = sample.ipu_util_percent
            rows.append(
                IpuStat(
                    index=index,
                    util=None if math.isnan(util) else float(util),
                    mem_used=int(memory.mem_used),
                    mem_total=int(memory.mem_total),
                    power=None if power_status != 0 else float(power.value),
                )
            )
        return rows, "; ".join(errors)

    def close(self) -> None:
        if self.ctx:
            self.lib.postmo_smi_shutdown(self.ctx)
            self.ctx = c_void_p()


def make_ipu_provider() -> IpuProvider | None:
    try:
        return IpuProvider()
    except (OSError, AttributeError):
        return None
