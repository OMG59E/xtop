"""GPU backend parsing and absence/error behavior."""

import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from xtop.gpu import NvidiaSmiProvider, NvmlProvider


class NvidiaSmiTest(unittest.TestCase):
    @patch("xtop.gpu.subprocess.run")
    def test_metrics_and_processes_need_only_two_calls(self, run):
        run.side_effect = [
            subprocess.CompletedProcess(
                [],
                0,
                "0, GPU-abc, Test GPU, 25, 10, 1024, 8192, 45, 80, 250\n",
                "",
            ),
            subprocess.CompletedProcess([], 0, "GPU-abc, 12345, python, 512\n", ""),
        ]
        provider = NvidiaSmiProvider("/usr/bin/nvidia-smi")

        gpus, error = provider.metrics()
        gpus, process_error = provider.processes(gpus)

        self.assertEqual(error, "")
        self.assertEqual(process_error, "")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(gpus[0].name, "Test GPU")
        self.assertEqual(gpus[0].mem_used, 1024)
        self.assertEqual(gpus[0].procs[0].pid, 12345)
        self.assertEqual(gpus[0].procs[0].mem_mib, 512)

    @patch("xtop.gpu.subprocess.run")
    def test_no_device_is_not_an_error(self, run):
        run.return_value = subprocess.CompletedProcess([], 6, "", "No devices were found")
        provider = NvidiaSmiProvider("/usr/bin/nvidia-smi")

        gpus, error = provider.metrics()

        self.assertEqual(gpus, [])
        self.assertEqual(error, "")


class _NvmlError(Exception):
    pass


class _FakeNvml:
    NVMLError = _NvmlError
    NVMLError_NotSupported = type("NotSupported", (_NvmlError,), {})
    NVML_TEMPERATURE_GPU = 0
    NVML_VALUE_NOT_AVAILABLE = -1

    def __init__(self):
        self.closed = False

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.closed = True

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return b"Test GPU"

    def nvmlDeviceGetUUID(self, handle):
        return "GPU-abc"

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=25, memory=10)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(used=1024 * 1024, total=8 * 1024 * 1024)

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 45

    def nvmlDeviceGetPowerUsage(self, handle):
        return 80_000

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return 250_000

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [SimpleNamespace(pid=12345, usedGpuMemory=512 * 1024 * 1024)]


class NvmlTest(unittest.TestCase):
    def test_units_identity_and_lifecycle(self):
        api = _FakeNvml()
        provider = NvmlProvider(api)

        gpus, error = provider.metrics()
        gpus, process_error = provider.processes(gpus)
        provider.close()

        self.assertEqual(error, "")
        self.assertEqual(process_error, "")
        self.assertEqual(gpus[0].uuid, "GPU-abc")
        self.assertEqual(gpus[0].mem_used, 1)
        self.assertEqual(gpus[0].power, 80)
        self.assertEqual(gpus[0].procs[0].mem_mib, 512)
        self.assertTrue(api.closed)


if __name__ == "__main__":
    unittest.main()
