"""Smoke-test the collector and layout math without a terminal."""

import time
import unittest
from types import SimpleNamespace

from xtop.collect import Collector, CpuMeter, GpuStat, _cpu_meter, _parse_stat_line
from xtop.layout import PRESETS, PRESET_ORDER
from xtop.monitor import Monitor
from xtop.ui import App, columns, compact, fixed, htop_mem, human, layout, pad, rate, spark


class FormatTest(unittest.TestCase):
    def test_human(self):
        self.assertEqual(human(512), "512B")
        self.assertEqual(human(1536), "1.5KB")
        self.assertEqual(rate(2 * 1024 * 1024), "2.0MB/s")
        self.assertEqual(htop_mem(1024), "1.0K")
        self.assertEqual(htop_mem(1024 * 1024), "1.0M")
        self.assertEqual(compact(1024 * 1024), "1.0M")

    def test_columns_resize(self):
        cells = [("DSK", 0), ("/data/very/long", 1), ("r 1.2MB/s", 0), ("698G", 0)]
        narrow = columns(cells, 40)
        wide = columns(cells, 100)
        self.assertLessEqual(len(narrow), 40)
        self.assertLessEqual(len(wide), 100)
        self.assertIn("/data", narrow)
        self.assertGreater(len(wide), len(narrow))
        # fixed columns stay the same width, the mount absorbs the growth
        self.assertTrue(narrow.endswith("698G"))
        self.assertTrue(wide.endswith("698G"))

    def test_pad_aligns_fixed_columns(self):
        lines = layout([[("PID", 0), ("1", 2)], [("PID", 0), ("624689", 2)]], 16)
        self.assertTrue(lines[0].endswith("     1"))
        self.assertTrue(lines[1].endswith("624689"))

    def test_layout_can_crop_without_ellipsis(self):
        line = layout([[("COMMAND", 0), ("long command", 1)]], 14, ellipsis=False)[0]
        self.assertEqual(line, "COMMAND long c")

    def test_fixed_summary_cell_has_no_ellipsis(self):
        self.assertEqual(fixed("very-long-device", 8), "very-lon")
        self.assertEqual(fixed("eth0", 8), "eth0    ")


class ParseTest(unittest.TestCase):
    def test_stat_line(self):
        line = "123 (my proc) S 1 1 1 0 -1 0 0 0 0 0 10 20 0 0 20 0 1 0 100 4096 5 0 0 0 0 0 0 0 0 0 0"
        self.assertEqual(_parse_stat_line(line)[:4], ("my proc", "S", 10, 20))
        self.assertEqual(_parse_stat_line(line)[4:], (20, 0, 4096))

    def test_cpu_delta(self):
        prev = [0] * 10
        cur = [40, 10, 20, 20, 10, 0, 0, 0, 0, 0]
        meter_obj = _cpu_meter(cur, prev)
        self.assertAlmostEqual(meter_obj.user, 40.0)
        self.assertAlmostEqual(meter_obj.nice, 10.0)
        self.assertAlmostEqual(meter_obj.system, 20.0)
        self.assertAlmostEqual(meter_obj.iowait, 10.0)
        self.assertAlmostEqual(meter_obj.busy, 70.0)

    def test_guest_is_not_double_counted(self):
        prev = [0] * 10
        cur = [50, 0, 0, 50, 0, 0, 0, 0, 30, 0]
        meter_obj = _cpu_meter(cur, prev)
        self.assertAlmostEqual(meter_obj.user, 20.0)
        self.assertAlmostEqual(meter_obj.guest, 30.0)
        self.assertAlmostEqual(meter_obj.busy, 50.0)

    def test_cpu_reset(self):
        self.assertEqual(_cpu_meter([1] + [0] * 9, [5] + [0] * 9), CpuMeter())


class LayoutTest(unittest.TestCase):
    def test_presets(self):
        for name in PRESET_ORDER:
            rows = PRESETS[name]
            self.assertTrue(rows)
            for height, cells in rows:
                self.assertGreaterEqual(height, 0)
                self.assertTrue(cells)
                for panel, weight in cells:
                    self.assertIn(panel, {"cpu", "mem", "gpu", "disk", "net", "fs", "proc"})
                    self.assertGreater(weight, 0)

    def test_wide_summary_puts_gpu_processes_on_the_right(self):
        gpu_proc = SimpleNamespace(
            gpu=0, pid=1234, user="alice", mem_mib=512, cmd="python train.py"
        )
        gpu = SimpleNamespace(
            index=0, util=75, mem_used=1024, mem_total=8192,
            temp=55, power=120, procs=[gpu_proc],
        )
        app = App.__new__(App)
        app.snap = SimpleNamespace(
            ready=True,
            mem=SimpleNamespace(used=1024, total=8192),
            cpu=SimpleNamespace(total=SimpleNamespace(busy=25)),
            nets=[], disks=[], gpus=[gpu], gpu_error="", ipus=[], ipu_error="",
        )

        narrow = app._rows(111, 24)
        edge = app._rows(112, 24)
        wide = app._rows(120, 24)

        self.assertFalse(any(" │ " in line for line in narrow))
        self.assertTrue(any(" │ " in line for line in edge))
        self.assertTrue(any(" │ " in line for line in wide))
        self.assertTrue(any("GPU0" in line for line in wide))
        self.assertTrue(any("COMMAND" in line for line in wide))
        self.assertTrue(any("python train.py" in line for line in wide))


class CollectorTest(unittest.TestCase):
    def test_two_samples(self):
        c = Collector()
        first = c.sample()
        self.assertFalse(first.ready)
        self.assertGreater(first.mem.total, 0)
        time.sleep(0.2)
        second = c.sample()
        self.assertTrue(second.ready)
        self.assertGreater(len(second.cpu.cores), 0)
        self.assertGreaterEqual(second.nproc, 1)
        self.assertGreaterEqual(second.threads, second.nproc)
        self.assertGreater(second.mem.total, second.mem.used)
        self.assertTrue(second.mounts)
        self.assertTrue(second.host)


class _FailingGpu:
    name = "fake"
    metric_interval = 0.05
    process_interval = 0.1

    def __init__(self):
        self.calls = 0
        self.closed = False

    def metrics(self):
        self.calls += 1
        if self.calls == 1:
            return [GpuStat(0, "fake", 10, 20, 30, 40, 50, 60, 70)], ""
        return [], "GPU lost"

    def processes(self, gpus):
        return gpus, ""

    def close(self):
        self.closed = True


class _CountingGpu:
    name = "fake"
    metric_interval = 0.04
    process_interval = 0.16

    def __init__(self):
        self.metric_calls = 0
        self.process_calls = 0

    def metrics(self):
        self.metric_calls += 1
        return [GpuStat(0, "fake", 10, 20, 30, 40, 50, 60, 70, "GPU-fake")], ""

    def processes(self, gpus):
        self.process_calls += 1
        return gpus, ""

    def close(self):
        pass


class MonitorTest(unittest.TestCase):
    def test_gpu_failure_keeps_last_good_snapshot(self):
        gpu = _FailingGpu()
        monitor = Monitor(0.2, gpu_provider=gpu)
        monitor.start()
        try:
            deadline = time.monotonic() + 2
            snap = None
            while time.monotonic() < deadline:
                snap = monitor.snapshot()
                if snap and snap.gpu_error:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(snap)
            self.assertEqual(snap.gpu_error, "GPU lost")
            self.assertEqual([item.name for item in snap.gpus], ["fake"])
        finally:
            monitor.close()
        self.assertTrue(gpu.closed)

    def test_gpu_metrics_and_processes_have_independent_frequencies(self):
        gpu = _CountingGpu()
        monitor = Monitor(0.2, gpu_provider=gpu)
        monitor.start()
        try:
            time.sleep(0.28)
        finally:
            monitor.close()
        self.assertGreaterEqual(gpu.metric_calls, 3)
        self.assertGreaterEqual(gpu.process_calls, 2)
        self.assertLess(gpu.process_calls, gpu.metric_calls)


if __name__ == "__main__":
    unittest.main()
