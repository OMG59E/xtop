"""Smoke-test the collector and layout math without a terminal."""

import time
import unittest

from xtop.collect import Collector, CpuMeter, _cpu_meter, _parse_stat_line
from xtop.layout import PRESETS, PRESET_ORDER
from xtop.ui import columns, compact, htop_mem, human, layout, pad, rate, spark


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
        self.assertTrue(any(p.user == "root" for p in second.procs))
        self.assertGreater(second.mem.total, second.mem.used)
        self.assertTrue(second.mounts)
        self.assertTrue(second.host)


if __name__ == "__main__":
    unittest.main()
