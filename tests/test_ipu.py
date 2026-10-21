"""Houmo IPU backend error and partial-result behavior."""

import ctypes
import unittest
from unittest.mock import patch

from xtop.ipu import IpuProvider, MAX_DEVICES, MIB, _Metrics, make_ipu_provider


def _write(pointer, ctype, value):
    ctypes.cast(pointer, ctypes.POINTER(ctype))[0] = ctype(value)


class _PartialApi:
    def postmo_smi_device_get_count(self, context, output):
        _write(output, ctypes.c_uint32, 2)
        return 0

    def postmo_smi_device_get_handle_by_index(self, context, index, output):
        _write(output, ctypes.c_void_p, index + 1)
        return 0

    def postmo_smi_device_get_metrics(self, device, output):
        index = device.value - 1
        if index == 1:
            return 9
        sample = ctypes.cast(output, ctypes.POINTER(_Metrics)).contents
        sample.ipu_util_percent = 25.0
        sample.mem_used_bytes = 1024 * MIB
        sample.mem_total_bytes = 8192 * MIB
        sample.power_w = 75.0
        return 0

    def postmo_smi_device_get_power(self, device, output):
        return 3

    def postmo_smi_shutdown(self, context):
        raise OSError("device already gone")


class _CountErrorApi(_PartialApi):
    def postmo_smi_device_get_count(self, context, output):
        raise RuntimeError("driver reset")


class _InvalidCountApi(_PartialApi):
    def postmo_smi_device_get_count(self, context, output):
        _write(output, ctypes.c_uint32, MAX_DEVICES + 1)
        return 0


class _UnavailableApi(_PartialApi):
    def postmo_smi_device_get_count(self, context, output):
        _write(output, ctypes.c_uint32, 1)
        return 0

    def postmo_smi_device_get_metrics(self, device, output):
        return 0


def _provider(api):
    provider = IpuProvider.__new__(IpuProvider)
    provider.lib = api
    provider.ctx = ctypes.c_void_p(123)
    return provider


class IpuProviderTest(unittest.TestCase):
    def test_partial_failure_keeps_healthy_device_and_fallback_fields(self):
        provider = _provider(_PartialApi())

        rows, error = provider.metrics()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].index, 0)
        self.assertEqual(rows[0].util, 25.0)
        self.assertEqual(rows[0].mem_used, 1024)
        self.assertEqual(rows[0].mem_total, 8192)
        self.assertEqual(rows[0].power, 75.0)
        self.assertEqual(error, "IPU1: metrics 9")

    def test_count_exception_becomes_error_result(self):
        rows, error = _provider(_CountErrorApi()).metrics()

        self.assertEqual(rows, [])
        self.assertIn("driver reset", error)

    def test_implausible_count_is_rejected(self):
        rows, error = _provider(_InvalidCountApi()).metrics()

        self.assertEqual(rows, [])
        self.assertIn("invalid device count", error)

    def test_empty_success_metrics_are_treated_as_not_ready(self):
        rows, error = _provider(_UnavailableApi()).metrics()

        self.assertEqual(rows, [])
        self.assertEqual(error, "IPU0: telemetry unavailable")

    def test_closed_provider_is_safe_and_shutdown_is_idempotent(self):
        provider = _provider(_PartialApi())

        provider.close()
        provider.close()
        rows, error = provider.metrics()

        self.assertEqual(rows, [])
        self.assertIn("not initialized", error)

    @patch("xtop.ipu.IpuProvider", side_effect=RuntimeError("bad ABI"))
    def test_factory_treats_sdk_or_abi_failure_as_no_provider(self, provider):
        self.assertIsNone(make_ipu_provider())


if __name__ == "__main__":
    unittest.main()
