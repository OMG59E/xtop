"""Terminal dashboard. Layout comes from layout.py."""

from __future__ import annotations

import argparse
import curses
import locale
import os
import sys
import time

from . import __version__
from .collect import Collector, Snapshot
from .layout import PRESET_ORDER, PRESETS

PANELS = ("cpu", "mem", "gpu", "disk", "net", "fs", "proc")
SORTS = ("cpu", "mem", "pid")


def htop_mem(n: float) -> str:
    """htop's memory text: KiB-based, one decimal, K/M/G/T."""
    units = ["K", "M", "G", "T", "P"]
    value = float(n) / 1024.0
    i = 0
    while abs(value) >= 1024 and i < len(units) - 1:
        value /= 1024.0
        i += 1
    return f"{value:.1f}{units[i]}"


def human(n: float, suffix: str = "B") -> str:
    units = ["", "K", "M", "G", "T", "P"]
    i = 0
    n = float(n)
    while abs(n) >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{n:.0f}{units[i]}{suffix}"
    return f"{n:.1f}{units[i]}{suffix}"


def rate(n: float) -> str:
    return human(n, "B/s")


def pct(n: float | None) -> str:
    if n is None:
        return "  n/a"
    return f"{n:5.1f}%"


def meter(segments: list[tuple[float, int]], width: int) -> list[tuple[str, int]]:
    """htop bar: each (ratio, color_pair) becomes that many colored blocks."""
    width = max(1, width)
    total = sum(max(0.0, ratio) for ratio, _ in segments)
    scale = min(1.0, total)
    if total > 1.0:
        segments = [(ratio / total * scale, attr) for ratio, attr in segments]
    cells: list[tuple[str, int]] = []
    used = 0
    for ratio, attr in segments:
        n = int(ratio * width)
        cells.extend(("|", attr) for _ in range(n))
        used += n
    cells.extend((" ", 0) for _ in range(width - used))
    return cells[:width]


def put_meter(win: curses.window, y: int, x: int, cells: list[tuple[str, int]], bracket: int) -> None:
    add(win, y, x, "[", bracket)
    col = x + 1
    h, w = win.getmaxyx()
    for ch, attr in cells:
        if col >= w - 1:
            break
        try:
            win.addch(y, col, ch, attr)
        except curses.error:
            break
        col += 1
    add(win, y, col, "]", bracket)


def spark(values: list[float], width: int, scale: float | None = None) -> str:
    ticks = " ▁▂▃▄▅▆▇█"
    if width <= 0:
        return ""
    data = values[-width:]
    if len(data) < width:
        data = [0.0] * (width - len(data)) + data
    peak = scale if scale and scale > 0 else max(data) or 1.0
    out = []
    for v in data:
        idx = int(min(8, max(0, round(v / peak * 8))))
        out.append(ticks[idx])
    return "".join(out)


def uptime(seconds: float) -> str:
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if d:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def sized(n: float) -> tuple[str, str]:
    """Split a byte count into a moving number and a fixed unit."""
    units = ["B", "K", "M", "G", "T", "P"]
    value = float(n)
    i = 0
    while abs(value) >= 1024 and i < len(units) - 1:
        value /= 1024.0
        i += 1
    number = f"{value:.0f}" if i == 0 or value >= 100 else f"{value:.1f}"
    return number, units[i]


def value(number: str, unit: str, width: int) -> str:
    """Fixed-width number with its unit glued on. The number is padded, never the unit."""
    gap = width - len(unit)
    return number.rjust(gap) + unit
def compact(n: float) -> str:
    number, unit = sized(n)
    return number + unit


def num(value: float, width: int, decimals: int = 1) -> str:
    """Right-align the moving part so the unit after it stays put."""
    text = f"{value:.{decimals}f}"
    return text.rjust(width)


def pad(rows: list[list[tuple[str, int]]]) -> list[list[tuple[str, int]]]:
    """Widen every column to its longest cell, keeping the alignment flag."""
    if not rows:
        return rows
    width = [0] * max(len(row) for row in rows)
    for row in rows:
        for i, (text, _) in enumerate(row):
            width[i] = max(width[i], len(text))
    padded = []
    for row in rows:
        cells = []
        for i, (text, grow) in enumerate(row):
            cells.append((text.ljust(width[i]) if grow != 2 else text, grow))
        padded.append(cells)
    return padded


def user_name(name: str) -> str:
    """top's USER column: 8 characters, longer names end with +."""
    if len(name) <= 8:
        return name
    return name[:7] + "+"
    return fit(text, width).ljust(width)


def columns(cells: list[tuple[str, int]], width: int, gap: int = 1) -> str:
    """cells are (text, grow). 0 keeps its width, 1 takes leftover, 2 is right aligned."""
    return layout([cells], width, gap)[0]


def layout(rows: list[list[tuple[str, int]]], width: int, gap: int = 1) -> list[str]:
    """Format rows together so a right-aligned column shares one width."""
    if not rows:
        return []
    count = max(len(row) for row in rows)
    col_w = [0] * count
    grow_of = [0] * count
    for row in rows:
        for i, (text, grow) in enumerate(row):
            col_w[i] = max(col_w[i], len(text))
            grow_of[i] = max(grow_of[i], grow)
    gaps = gap * (count - 1)
    spare = width - gaps - sum(1 if grow_of[i] == 1 else col_w[i] for i in range(count))
    flex = [i for i in range(count) if grow_of[i] == 1]
    if spare < 0 and flex:
        for i in flex:
            if spare >= 0:
                break
            cut = min(col_w[i] - 1, -spare)
            col_w[i] -= max(0, cut)
            spare += max(0, cut)
        spare = max(0, spare)
    if flex and spare > 0:
        share, rest = divmod(spare, len(flex))
        for n, i in enumerate(flex):
            col_w[i] += share + (1 if n < rest else 0)
    lines = []
    for row in rows:
        parts = []
        for i in range(len(row)):
            text, grow = row[i]
            shown = fit(text, col_w[i])
            parts.append(shown.rjust(col_w[i]) if grow == 2 else shown.ljust(col_w[i]))
        lines.append((gap * " ").join(parts)[:width])
    return lines


def level_attr(ratio: float) -> int:
    if ratio >= 0.90:
        return curses.color_pair(3) | curses.A_BOLD
    if ratio >= 0.70:
        return curses.color_pair(2)
    return curses.color_pair(1)


def add(win: curses.window, y: int, x: int, text: str, attr: int = 0, width: int | None = None) -> None:
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    limit = w - x - 1
    if width is not None:
        limit = min(limit, width)
    if limit <= 0:
        return
    try:
        win.addnstr(y, x, text, limit, attr)
    except curses.error:
        pass


class App:
    def __init__(self, interval: float, preset: str) -> None:
        self.interval = interval
        self.collector = Collector()
        self.preset = preset if preset in PRESETS else PRESET_ORDER[0]
        self.sort = "cpu"
        self.proc_offset = 0
        self.paused = False
        self.snap: Snapshot | None = None
        self.help = False

    def run(self, stdscr: curses.window) -> None:
        curses.curs_set(0)
        curses.use_default_colors()
        stdscr.nodelay(True)
        stdscr.keypad(True)
        self._colors()
        next_sample = 0.0
        while True:
            now = time.monotonic()
            if not self.paused and now >= next_sample:
                self.snap = self.collector.sample()
                next_sample = now + self.interval
            self.draw(stdscr)
            key = self._wait_key(stdscr, max(0.05, next_sample - time.monotonic()))
            if key is None:
                continue
            if not self._handle(key):
                return

    def _colors(self) -> None:
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_BLUE, -1)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)
        curses.init_pair(7, curses.COLOR_WHITE, -1)
        curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(11, curses.COLOR_BLACK, curses.COLOR_WHITE)

    def _wait_key(self, stdscr: curses.window, timeout: float) -> int | None:
        stdscr.timeout(max(1, int(timeout * 1000)))
        try:
            key = stdscr.getch()
        except curses.error:
            return None
        if key == -1:
            return None
        return key

    def _handle(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return False
        if key in (ord("?"), ord("h")):
            self.help = not self.help
            return True
        if self.help and key in (ord(" "), 10, curses.KEY_ENTER):
            self.help = False
            return True
        if key == ord(" "):
            self.paused = not self.paused
        elif key == ord("l"):
            idx = PRESET_ORDER.index(self.preset) if self.preset in PRESET_ORDER else -1
            self.preset = PRESET_ORDER[(idx + 1) % len(PRESET_ORDER)]
        elif key == ord("s"):
            self.sort = SORTS[(SORTS.index(self.sort) + 1) % len(SORTS)]
            self.proc_offset = 0
        elif key == ord("r"):
            self.snap = self.collector.sample()
        elif key == curses.KEY_DOWN:
            self.proc_offset += 1
        elif key == curses.KEY_UP:
            self.proc_offset = max(0, self.proc_offset - 1)
        elif key == curses.KEY_NPAGE:
            self.proc_offset += 10
        elif key == curses.KEY_PPAGE:
            self.proc_offset = max(0, self.proc_offset - 10)
        elif key == curses.KEY_RESIZE:
            pass
        return True

    def draw(self, stdscr: curses.window) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 6 or w < 40:
            add(stdscr, 0, 0, "terminal too small", curses.color_pair(3))
            stdscr.refresh()
            return
        self._header(stdscr, w)
        rows = self._device_rows(w)
        for i, (text, attr) in enumerate(rows):
            add(stdscr, 1 + i, 0, text, attr, w - 1)
        top = 1 + len(rows)
        self._layout(stdscr, top, h - top, w)
        if self.help:
            self._help(stdscr, h, w)
        stdscr.refresh()

    def _header(self, stdscr: curses.window, w: int) -> None:
        snap = self.snap
        if not snap or not snap.ready:
            add(stdscr, 0, 0, " xtop  collecting…".ljust(w - 1), curses.color_pair(8) | curses.A_BOLD)
            return
        load = "/".join(f"{v:.1f}" for v in snap.load)
        title = f" xtop {snap.host} up {uptime(snap.uptime)} load {load} tasks {snap.nproc}"
        add(stdscr, 0, 0, title.ljust(w - 1), curses.color_pair(8) | curses.A_BOLD)

    def _device_rows(self, w: int) -> list[tuple[str, int]]:
        snap = self.snap
        if snap is None or not snap.ready:
            return [(" collecting…", curses.color_pair(7))]
        plain = curses.color_pair(7)
        mem = snap.mem
        mem_pct = 100.0 * mem.used / mem.total if mem.total else 0.0
        specs: list[tuple[list[tuple[str, int]], int]] = []
        used_n, used_u = sized(mem.used)
        total_n, total_u = sized(mem.total)
        specs.append((
            [
                ("CPU", 0), (value(num(snap.cpu.total.busy, 5).strip(), "%", 6), 0),
                ("MEM", 0), (value(num(mem_pct, 5).strip(), "%", 6), 0),
                (value(used_n, used_u, 6), 0), ("/", 0), (value(total_n, total_u, 6), 1),
            ],
            plain,
        ))
        if snap.gpu_error:
            specs.append(([("GPU", 0), (snap.gpu_error, 1)], curses.color_pair(3)))
        for gpu in snap.gpus:
            used_n, used_u = sized((gpu.mem_used or 0.0) * 1024 * 1024)
            total_n, total_u = sized((gpu.mem_total or 0.0) * 1024 * 1024)
            pct = 100.0 * (gpu.mem_used or 0.0) / gpu.mem_total if gpu.mem_total else 0.0
            temp = f"{gpu.temp:.0f}" if gpu.temp is not None else ""
            power = f"{gpu.power:.0f}" if gpu.power is not None else ""
            specs.append((
                [
                    (f"GPU{gpu.index}", 0), (value(num(gpu.util or 0, 5).strip(), "%", 6), 0),
                    ("MEM", 0), (value(num(pct, 5).strip(), "%", 6), 0),
                    (value(used_n, used_u, 6), 0), ("/", 0), (value(total_n, total_u, 6), 0),
                    (value(temp, "C", 4), 0), (value(power, "W", 5), 1),
                ],
                plain,
            ))
        for net in snap.nets:
            rx_n, rx_u = sized(net.rx_bps)
            tx_n, tx_u = sized(net.tx_bps)
            specs.append((
                [
                    ("NET", 0), (net.name, 0),
                    ("rx", 0), (value(rx_n, rx_u + "/s", 9), 0),
                    ("tx", 0), (value(tx_n, tx_u + "/s", 9), 1),
                ],
                plain,
            ))
        disks = []
        for disk in snap.disks:
            name = disk.mounts.split(",")[0] if disk.mounts else disk.name
            free_n, free_u = sized(disk.free) if disk.total else ("-", "")
            total_n, total_u = sized(disk.total) if disk.total else ("-", "")
            read_n, read_u = sized(disk.read_bps)
            write_n, write_u = sized(disk.write_bps)
            disks.append([
                ("DSK", 0), (name, 1),
                ("r", 0), (value(read_n, read_u + "/s", 9), 0),
                ("w", 0), (value(write_n, write_u + "/s", 9), 0),
                (value(free_n, free_u, 6), 0), (value(total_n, total_u, 6), 0),
            ])
        widest = [0] * len(disks[0]) if disks else []
        for row in disks:
            for i, (text, _) in enumerate(row):
                widest[i] = max(widest[i], len(text))
        fixed = sum(widest[i] for i in range(7) if i != 1) + 6
        mount_w = max(1, (w - 1) - fixed)
        for row in disks:
            row[1] = (fit(row[1][0], mount_w), 0)
        specs.extend((row, plain) for row in pad(disks))
        return [(columns(cells, w - 1), attr) for cells, attr in specs]

    def _layout(self, stdscr: curses.window, top: int, height: int, width: int) -> None:
        rows = PRESETS[self.preset]
        fixed = sum(h for h, _ in rows if h > 0)
        flex = [i for i, (h, _) in enumerate(rows) if h <= 0]
        leftover = height - fixed
        if leftover < len(flex):
            # not enough rows: shrink fixed rows from the bottom
            rows = [row for row in rows if row[0] <= 0 or True]
        base = leftover // len(flex) if flex else 0
        extra = leftover - base * len(flex) if flex else 0
        if base < 3 and flex:
            base = 3
        y = top
        for i, (row_h, cells) in enumerate(rows):
            h = row_h if row_h > 0 else base + (extra if i == flex[-1] else 0)
            if y >= top + height or h <= 0:
                break
            h = min(h, top + height - y)
            if h < 3:
                continue
            weights = [max(1, weight) for _, weight in cells]
            total_w = sum(weights)
            x = 0
            for j, ((name, _), weight) in enumerate(zip(cells, weights)):
                cw = width - x if j == len(cells) - 1 else max(8, width * weight // total_w)
                cw = min(cw, width - x)
                if cw >= 8:
                    self._panel(stdscr, y, x, h, cw, name)
                x += cw
            y += h

    def _panel(self, stdscr: curses.window, y: int, x: int, h: int, w: int, name: str) -> None:
        if h < 1 or w < 16:
            return
        try:
            win = stdscr.derwin(h, w, y, x)
        except curses.error:
            return
        win.erase()
        snap = self.snap
        if snap is None or not snap.ready:
            add(win, 0, 0, "collecting…")
            return
        drawer = {"cpu": self._draw_cpu, "mem": self._draw_mem, "proc": self._draw_proc}.get(name)
        if drawer:
            drawer(win, h, w, snap)

    def _cpu_bar(self, meter_obj, width: int) -> list[tuple[str, int]]:
        # htop default order: nice system irq softirq steal guest io user
        parts = [
            (meter_obj.nice / 100, curses.color_pair(2)),
            (meter_obj.system / 100, curses.color_pair(3)),
            (meter_obj.irq / 100, curses.color_pair(4)),
            (meter_obj.softirq / 100, curses.color_pair(5)),
            (meter_obj.steal / 100, curses.color_pair(6)),
            (meter_obj.guest / 100, curses.color_pair(9)),
            (meter_obj.iowait / 100, curses.color_pair(10)),
            (meter_obj.user / 100, curses.color_pair(1)),
        ]
        return meter(parts, width)

    def _draw_cpu(self, win: curses.window, h: int, width: int, snap: Snapshot) -> None:
        cpu = snap.cpu
        rows = [cpu.total, *cpu.cores]
        label_w = 4 if len(cpu.cores) < 100 else 5
        text_w = 7
        bar_w = max(4, width - label_w - text_w - 3)
        for i, item in enumerate(rows):
            if i >= h:
                break
            label = "CPU" if i == 0 else str(i - 1)
            add(win, i, 0, f"{label:>{label_w}}", curses.color_pair(4))
            put_meter(win, i, label_w + 1, self._cpu_bar(item, bar_w), curses.color_pair(1))
            add(win, i, label_w + bar_w + 3, f"{item.busy:5.1f}%", curses.color_pair(7))

    def _mem_bar(self, mem, width: int) -> list[tuple[str, int]]:
        total = mem.total or 1
        parts = [
            (mem.used / total, curses.color_pair(1)),
            (mem.buffers / total, curses.color_pair(2)),
            (mem.shared / total, curses.color_pair(3)),
            (mem.cache / total, curses.color_pair(6)),
        ]
        return meter(parts, width)

    def _swap_bar(self, mem, width: int) -> list[tuple[str, int]]:
        total = mem.swap_total or 1
        parts = [
            (mem.swap_used / total, curses.color_pair(3)),
            (mem.swap_cache / total, curses.color_pair(6)),
        ]
        return meter(parts, width)

    def _draw_mem(self, win: curses.window, h: int, width: int, snap: Snapshot) -> None:
        mem = snap.mem
        label_w = 4
        text = f"{htop_mem(mem.used)}/{htop_mem(mem.total)}"
        bar_w = max(4, width - label_w - len(text) - 4)
        if h >= 1:
            add(win, 0, 0, " Mem", curses.color_pair(4))
            put_meter(win, 0, label_w + 1, self._mem_bar(mem, bar_w), curses.color_pair(1))
            add(win, 0, label_w + bar_w + 3, text, curses.color_pair(7))
        if h >= 2:
            swap = f"{htop_mem(mem.swap_used)}/{htop_mem(mem.swap_total)}"
            add(win, 1, 0, " Swp", curses.color_pair(4))
            put_meter(win, 1, label_w + 1, self._swap_bar(mem, bar_w), curses.color_pair(3))
            add(win, 1, label_w + bar_w + 3, swap, curses.color_pair(7))

    def _draw_proc(self, win: curses.window, h: int, width: int, snap: Snapshot) -> None:
        key = {"cpu": lambda p: p.cpu, "mem": lambda p: p.rss, "pid": lambda p: -p.pid}[self.sort]
        procs = sorted(snap.procs, key=key, reverse=True)
        visible = max(0, h - 1)
        if self.proc_offset > max(0, len(procs) - visible):
            self.proc_offset = max(0, len(procs) - visible)
        shown = procs[self.proc_offset : self.proc_offset + visible]
        table = [[
            ("PID", 2), (user_name("USER"), 0), ("%CPU", 2), ("%MEM", 2), ("RSS", 2), ("CMD", 1),
        ]]
        for proc in shown:
            rss_n, rss_u = sized(proc.rss)
            table.append([
                (str(proc.pid), 2), (user_name(proc.user), 0),
                (f"{proc.cpu:5.1f}", 2),
                (f"{proc.mem:5.1f}", 2),
                (value(rss_n, rss_u, 6), 2),
                (proc.cmd, 1),
            ])
        table = pad(table)
        lines = layout(table, width - 1)
        add(win, 0, 0, lines[0], curses.A_BOLD)
        for i, (proc, line) in enumerate(zip(shown, lines[1:])):
            add(win, 1 + i, 0, line, curses.color_pair(3) if proc.cpu >= 50 else 0)

    def _help(self, stdscr: curses.window, h: int, w: int) -> None:
        lines = [
            "xtop",
            "",
            "布局写在 xtop/layout.py",
            "一行 = (高度, [(面板, 宽度权重), ...])",
            "高度 0 = 吃掉剩余空间",
            "面板: cpu mem gpu disk net fs proc",
            "",
            "q / esc   退出",
            "l         切换预设布局",
            "s         进程排序 cpu / mem / pid",
            "space     暂停",
            "r         立刻刷新",
            "↑ ↓       滚动进程",
            "?         关闭帮助",
        ]
        bh, bw = len(lines) + 2, 46
        y = max(1, (h - bh) // 2)
        x = max(1, (w - bw) // 2)
        try:
            win = stdscr.derwin(bh, min(bw, w - 2), y, x)
        except curses.error:
            return
        win.erase()
        win.box()
        for i, line in enumerate(lines):
            add(win, 1 + i, 2, line, curses.A_BOLD if i == 0 else 0)


def main(argv: list[str] | None = None) -> int:
    locale.setlocale(locale.LC_ALL, "")
    parser = argparse.ArgumentParser(prog="xtop", description="small resource monitor")
    parser.add_argument("-d", "--delay", type=float, default=2.0, help="sample interval seconds")
    parser.add_argument("-l", "--layout", default=PRESET_ORDER[0], choices=list(PRESETS), help="layout preset")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if args.delay <= 0:
        parser.error("--delay must be > 0")
    if not sys.stdout.isatty():
        print("xtop needs a terminal", file=sys.stderr)
        return 1
    os.environ.setdefault("ESCDELAY", "25")
    app = App(args.delay, args.layout)
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
