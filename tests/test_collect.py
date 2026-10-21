"""Smoke-test the collector and layout math without a terminal."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from xtop.collect import Collector, CpuMeter, GpuStat, IpuStat, _cpu_meter, _parse_stat_line
from xtop.layout import PRESETS, PRESET_ORDER
from xtop.monitor import Monitor
from xtop.ui import (
    WIDE_SUMMARY_MIN_WIDTH,
    App,
    columns,
    compact,
    fixed,
    htop_mem,
    human,
    layout,
    pad,
    rate,
    spark,
    summary_left_width,
)


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

        narrow = app._rows(WIDE_SUMMARY_MIN_WIDTH - 1, 24)
        edge = app._rows(WIDE_SUMMARY_MIN_WIDTH, 24)
        wide = app._rows(WIDE_SUMMARY_MIN_WIDTH + 8, 24)

        self.assertFalse(any(" │ " in line for line in narrow))
        self.assertTrue(any(" │ " in line for line in edge))
        self.assertTrue(any(" │ " in line for line in wide))
        self.assertTrue(any("GPU0" in line for line in wide))
        self.assertTrue(any("COMMAND" in line for line in wide))
        self.assertTrue(any("python train.py" in line for line in wide))
        self.assertTrue(all(line.index(" │ ") == 40 for line in edge))

    def test_wide_summary_divider_tracks_left_content_with_limits(self):
        self.assertEqual(summary_left_width(["short"]), 40)
        self.assertEqual(summary_left_width(["x" * 54]), 54)
        self.assertEqual(summary_left_width(["x" * 100]), 62)

    def test_network_and_mount_share_current_width_capped_at_19(self):
        app = App.__new__(App)
        app.snap = SimpleNamespace(
            ready=True,
            mem=SimpleNamespace(used=1024, total=8192),
            cpu=SimpleNamespace(total=SimpleNamespace(busy=25)),
            nets=[SimpleNamespace(name="10.64.35.39", rx_bps=1, tx_bps=2)],
            disks=[SimpleNamespace(
                name="sda", mounts="/data04", read_bps=3,
                write_bps=4, free=5, total=6,
            )],
            gpus=[], gpu_error="", ipus=[], ipu_error="",
        )
        app.show_disk_rate = True

        sections = dict(app._section_lines())
        net = sections["net"][0]
        disk = sections["disk"][0]

        self.assertEqual(net[4:15], "10.64.35.39")
        self.assertEqual(net[15:18], "  ↑")
        self.assertLess(net.index("2B/s", 18), net.index("1B/s", 18))
        self.assertEqual(disk[4:15], "/data04".ljust(11))
        self.assertEqual(disk[15:18], "  r")

        app.snap.disks[0].mounts = "/12345678901234567890"
        sections = dict(app._section_lines())
        net = sections["net"][0]
        disk = sections["disk"][0]

        self.assertEqual(net[4:23], "10.64.35.39".ljust(19))
        self.assertEqual(net[23:26], "  ↑")
        self.assertEqual(disk[4:23], "/123456789012345678")
        self.assertEqual(disk[23:26], "  r")

    def test_disk_rates_are_hidden_by_default(self):
        app = App.__new__(App)
        app.snap = SimpleNamespace(disks=[SimpleNamespace(
            name="sda", mounts="/data", read_bps=1024, write_bps=2048,
            free=3 * 1024**3, total=4 * 1024**3,
        )])

        line = app._disk_lines(5)[0]

        self.assertEqual(line, "DSK /data    3.0G   4.0G")
        self.assertNotIn("/s", line)


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

    def test_disks_are_sorted_by_mount_name(self):
        collector = Collector()
        collector.disk_order = {"sda": 0, "sdb": 1, "sdc": 2}
        collector.disk_mounts = {
            "sda": ["/zeta"],
            "sdb": ["/alpha"],
            "sdc": ["/middle"],
        }
        collector.disk_usage = {
            name: (1024, 512) for name in collector.disk_order
        }
        collector.prev_disk = {
            name: (0, 0, 0) for name in collector.disk_order
        }
        counters = {
            name: (1024, 2048, 10) for name in collector.disk_order
        }

        with patch("xtop.collect._diskstats", return_value=counters):
            disks = collector._disks(1.0, refresh_topology=False)

        self.assertEqual([disk.mounts for disk in disks], ["/alpha", "/middle", "/zeta"])


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


class _PartialIpu:
    interval = 0.04

    def __init__(self):
        self.calls = 0
        self.closed = False

    def metrics(self):
        self.calls += 1
        if self.calls == 1:
            return [IpuStat(0, 10, 100, 1000, 20), IpuStat(1, 11, 200, 1000, 21)], ""
        return [IpuStat(0, 30, 300, 1000, 23)], "IPU1: metrics 9"

    def close(self):
        self.closed = True


class MonitorTest(unittest.TestCase):
    def test_gpu_failure_keeps_last_good_snapshot(self):
        gpu = _FailingGpu()
        monitor = Monitor(0.2, gpu_provider=gpu, ipu_provider=None)
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
        monitor = Monitor(0.2, gpu_provider=gpu, ipu_provider=None)
        monitor.start()
        try:
            time.sleep(0.28)
        finally:
            monitor.close()
        self.assertGreaterEqual(gpu.metric_calls, 3)
        self.assertGreaterEqual(gpu.process_calls, 2)
        self.assertLess(gpu.process_calls, gpu.metric_calls)

    def test_ipu_partial_failure_updates_healthy_and_retains_failed_device(self):
        ipu = _PartialIpu()
        monitor = Monitor(0.2, gpu_provider=_CountingGpu(), ipu_provider=ipu)
        monitor.start()
        try:
            deadline = time.monotonic() + 2
            snap = None
            while time.monotonic() < deadline:
                snap = monitor.snapshot()
                if snap and snap.ipu_error:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(snap)
            self.assertEqual(snap.ipu_error, "IPU1: metrics 9")
            self.assertEqual([item.index for item in snap.ipus], [0, 1])
            self.assertEqual(snap.ipus[0].util, 30)
            self.assertEqual(snap.ipus[1].util, 11)
        finally:
            monitor.close()
        self.assertTrue(ipu.closed)


if __name__ == "__main__":
    unittest.main()
