"""Terminal dashboard. Layout comes from layout.py."""

from __future__ import annotations

import argparse
import curses
import locale
import os
import sys

from . import __version__
from .collect import Snapshot
from .layout import PRESET_ORDER, PRESETS
from .monitor import Monitor

PANELS = ("cpu", "mem", "gpu", "disk", "net", "fs", "proc")
SORTS = ("cpu", "mem", "pid")
SUMMARY_NET_WIDTH = 12
SUMMARY_MOUNT_WIDTH = 16
SUMMARY_GPU_WIDTH = 5
SUMMARY_LEFT_WIDTH = 60
GPU_PANEL_MIN_WIDTH = 48
SUMMARY_SEPARATOR = " │ "
WIDE_SUMMARY_MIN_WIDTH = (
    SUMMARY_LEFT_WIDTH + len(SUMMARY_SEPARATOR) + GPU_PANEL_MIN_WIDTH + 1
)
PROC_HEADER_PAIR = 1


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
    """Build a text meter from (ratio, terminal attribute) segments."""
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


def fit(text: str, width: int, ellipsis: bool = True) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if not ellipsis or width <= 1:
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


def fixed(text: str, width: int) -> str:
    """A fixed-width summary cell, cropped without an ellipsis."""
    return text[:width].ljust(width)


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


def box(title: str, lines: list[str], width: int) -> list[str]:
    """Round-cornered box. Content wider than the box is cut at the right edge."""
    inner = max(1, width - 2)
    top = "╭" + title[:inner].center(inner, "─") + "╮"
    body = ["│" + fit(line, inner).ljust(inner) + "│" for line in lines]
    bottom = "╰" + "─" * inner + "╯"
    return [top, *body, bottom]


def user_name(name: str) -> str:
    if len(name) <= 8:
        return name
    return name[:7] + "+"


def columns(cells: list[tuple[str, int]], width: int, gap: int = 1) -> str:
    """cells are (text, grow). 0 keeps its width, 1 takes leftover, 2 is right aligned."""
    return layout([cells], width, gap)[0]


def layout(
    rows: list[list[tuple[str, int]]],
    width: int,
    gap: int = 1,
    ellipsis: bool = True,
) -> list[str]:
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
    flex = [i for i in range(count) if grow_of[i] == 1]
    if flex:
        # Flexible columns own exactly the space left by fixed columns. Their
        # natural text width must not be counted twice when distributing it.
        fixed = sum(col_w[i] for i in range(count) if i not in flex)
        available = max(len(flex), width - gaps - fixed)
        share, rest = divmod(available, len(flex))
        for n, i in enumerate(flex):
            col_w[i] = share + (1 if n < rest else 0)
    lines = []
    for row in rows:
        parts = []
        for i in range(len(row)):
            text, grow = row[i]
            shown = fit(text, col_w[i], ellipsis)
            parts.append(shown.rjust(col_w[i]) if grow == 2 else shown.ljust(col_w[i]))
        lines.append((gap * " ").join(parts)[:width])
    return lines


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
    def __init__(self, interval: float, preset: str, refresh: float = 3.0) -> None:
        self.interval = interval
        self.refresh_interval = refresh
        self.monitor = Monitor(interval)
        self.preset = preset if preset in PRESETS else PRESET_ORDER[0]
        self.sort = "cpu"
        self.proc_offset = 0
        self.paused = False
        self.snap: Snapshot | None = None
        self.help = False
        self.proc_header_attr = curses.A_REVERSE
        self.refresh_pending = False

    def run(self, stdscr: curses.window) -> None:
        curses.curs_set(0)
        # Pair 0 follows the terminal's own foreground/background defaults.
        curses.use_default_colors()
        self._init_styles()
        stdscr.nodelay(True)
        stdscr.keypad(True)
        self.monitor.start()
        try:
            while True:
                if not self.paused:
                    self.snap = self.monitor.snapshot()
                self.draw(stdscr)
                timeout = 0.1 if self.refresh_pending else self.refresh_interval
                self.refresh_pending = False
                key = self._wait_key(stdscr, timeout)
                if key is not None and not self._handle(key):
                    return
        finally:
            self.monitor.close()

    def _init_styles(self) -> None:
        """Use a top-like process header, with reverse video as fallback."""
        self.proc_header_attr = curses.A_REVERSE
        if not curses.has_colors():
            return
        try:
            curses.init_pair(PROC_HEADER_PAIR, curses.COLOR_BLACK, curses.COLOR_WHITE)
            self.proc_header_attr = curses.color_pair(PROC_HEADER_PAIR)
        except curses.error:
            pass

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
            self.monitor.refresh()
            self.refresh_pending = True
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
            add(stdscr, 0, 0, "terminal too small")
            stdscr.refresh()
            return
        self._header(stdscr, w)
        rows = self._rows(w, h)
        for i, line in enumerate(rows):
            add(stdscr, 1 + i, 0, line, width=w - 1)
        top = 1 + len(rows)
        self._layout(stdscr, top, h - top, w)
        if self.help:
            self._help(stdscr, h, w)
        stdscr.refresh()

    def _header(self, stdscr: curses.window, w: int) -> None:
        snap = self.snap
        if not snap or not snap.ready:
            add(stdscr, 0, 0, " xtop  collecting…".ljust(w - 1), curses.A_BOLD)
            return
        load = "/".join(f"{v:.1f}" for v in snap.load)
        title = f" xtop {snap.host} up {uptime(snap.uptime)} load {load} tasks {snap.nproc}"
        if snap.errors:
            title += " !" + ",".join(sorted(snap.errors))
        add(stdscr, 0, 0, title.ljust(w - 1), curses.A_BOLD)

    def _rows(self, width: int, height: int) -> list[str]:
        sections = self._section_lines()
        snap = self.snap
        if not snap or not snap.ready or not snap.gpus or width < WIDE_SUMMARY_MIN_WIDTH:
            return [line for _, group in sections for line in group]

        left = [line for name, group in sections if name != "gpu" for line in group]
        max_gpu_rows = max(4, min(10, height // 3))
        right = self._gpu_panel_lines(max_gpu_rows)
        right_width = width - SUMMARY_LEFT_WIDTH - len(SUMMARY_SEPARATOR) - 1
        rows = []
        for index in range(max(len(left), len(right))):
            left_line = left[index] if index < len(left) else ""
            right_line = right[index] if index < len(right) else ""
            rows.append(
                fixed(left_line, SUMMARY_LEFT_WIDTH)
                + SUMMARY_SEPARATOR
                + right_line[:right_width]
            )
        return rows

    def _section_lines(self) -> list[tuple[str, list[str]]]:
        snap = self.snap
        if snap is None or not snap.ready:
            return [("cpu", ["collecting…"])]
        mem = snap.mem
        mem_pct = 100.0 * mem.used / mem.total if mem.total else 0.0
        used_n, used_u = sized(mem.used)
        total_n, total_u = sized(mem.total)
        host = (
            f"CPU {value(num(snap.cpu.total.busy, 5).strip(), '%', 6)} "
            f"MEM {value(num(mem_pct, 5).strip(), '%', 6)} "
            f"{value(used_n, used_u, 6)}/{value(total_n, total_u, 6)}"
        )
        net_lines = []
        for net in snap.nets:
            rx_n, rx_u = sized(net.rx_bps)
            tx_n, tx_u = sized(net.tx_bps)
            net_lines.append(
                f"NET {fixed(net.name, SUMMARY_NET_WIDTH)} rx {value(rx_n, rx_u + '/s', 9)} "
                f"tx {value(tx_n, tx_u + '/s', 9)}"
            )
        disk_lines = self._disk_lines()
        gpu_lines = self._gpu_summary_lines()
        if snap.gpu_error:
            gpu_lines.insert(0, snap.gpu_error)
        return [
            ("cpu", [host]),
            ("net", net_lines or ["NET none"]),
            ("disk", disk_lines or ["DSK none"]),
            ("gpu", gpu_lines or ["GPU none"]),
        ]

    def _gpu_summary_lines(self) -> list[str]:
        snap = self.snap
        if snap is None:
            return []
        lines = []
        for gpu in snap.gpus:
            used_n, used_u = sized((gpu.mem_used or 0.0) * 1024 * 1024)
            total_n, total_u = sized((gpu.mem_total or 0.0) * 1024 * 1024)
            temp = f"{gpu.temp:.0f}" if gpu.temp is not None else ""
            power = f"{gpu.power:.0f}" if gpu.power is not None else ""
            lines.append(
                f"{fixed(f'GPU{gpu.index}', SUMMARY_GPU_WIDTH)} "
                f"{value(num(gpu.util or 0, 5).strip(), '%', 6)} "
                f"{value(used_n, used_u, 6)}/{value(total_n, total_u, 6)} "
                f"{value(temp, 'C', 4)} {value(power, 'W', 5)}"
            )
        return lines

    def _gpu_panel_lines(self, max_rows: int) -> list[str]:
        snap = self.snap
        if snap is None:
            return []
        lines = self._gpu_summary_lines()
        if snap.gpu_error:
            lines.insert(0, snap.gpu_error)

        processes = [proc for gpu in snap.gpus for proc in gpu.procs]
        processes.sort(key=lambda proc: (proc.gpu, -(proc.mem_mib or 0), proc.pid))
        if not processes:
            if len(lines) < max_rows:
                lines.append("no GPU processes")
            return lines

        if len(lines) >= max_rows:
            return lines
        header = f"{'GPU':>3} {'PID':>7} {'USER':<8} {'MEM':>7} COMMAND"
        lines.append(header)
        room = max(0, max_rows - len(lines))
        shown = len(processes) if len(processes) <= room else max(0, room - 1)
        for proc in processes[:shown]:
            if proc.mem_mib is None:
                mem = value("-", "", 7)
            else:
                mem_n, mem_u = sized(proc.mem_mib * 1024 * 1024)
                mem = value(mem_n, mem_u, 7)
            lines.append(
                f"{proc.gpu:>3} {proc.pid:>7} {fixed(proc.user, 8)} {mem} {proc.cmd}"
            )
        hidden = len(processes) - shown
        if hidden > 0 and room > 0:
            lines.append(f"+{hidden} more")
        return lines

    def _disk_lines(self) -> list[str]:
        snap = self.snap
        lines = []
        for disk in snap.disks:
            name = disk.mounts.split(",")[0] if disk.mounts else disk.name
            free_n, free_u = sized(disk.free) if disk.total else ("-", "")
            total_n, total_u = sized(disk.total) if disk.total else ("-", "")
            read_n, read_u = sized(disk.read_bps)
            write_n, write_u = sized(disk.write_bps)
            lines.append(
                f"DSK {fixed(name, SUMMARY_MOUNT_WIDTH)} r {value(read_n, read_u + '/s', 9)} "
                f"w {value(write_n, write_u + '/s', 9)} "
                f"{value(free_n, free_u, 6)} {value(total_n, total_u, 6)}"
            )
        return lines

    def _layout(self, stdscr: curses.window, top: int, height: int, width: int) -> None:
        rows = PRESETS[self.preset]
        flex = [i for i, (h, _) in enumerate(rows) if h <= 0]
        heights = [0] * len(rows)

        # Keep room for flexible panels. When the terminal is short, fixed
        # panels at the bottom are shrunk or dropped first.
        fixed_budget = max(0, height - 3 * len(flex))
        for i, (requested, _) in enumerate(rows):
            if requested <= 0:
                continue
            assigned = min(requested, fixed_budget)
            heights[i] = assigned if assigned >= 3 else 0
            fixed_budget -= heights[i]

        leftover = max(0, height - sum(heights))
        if flex:
            base, extra = divmod(leftover, len(flex))
            for n, i in enumerate(flex):
                heights[i] = base + (1 if n < extra else 0)

        y = top
        for i, (_, cells) in enumerate(rows):
            h = heights[i]
            if y >= top + height or h <= 0:
                continue
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
        return meter([(meter_obj.busy / 100, 0)], width)

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
            add(win, i, 0, f"{label:>{label_w}}")
            put_meter(win, i, label_w + 1, self._cpu_bar(item, bar_w), 0)
            add(win, i, label_w + bar_w + 3, f"{item.busy:5.1f}%")

    def _mem_bar(self, mem, width: int) -> list[tuple[str, int]]:
        total = mem.total or 1
        shown = mem.used + mem.buffers + mem.shared + mem.cache
        return meter([(shown / total, 0)], width)

    def _swap_bar(self, mem, width: int) -> list[tuple[str, int]]:
        total = mem.swap_total or 1
        return meter([((mem.swap_used + mem.swap_cache) / total, 0)], width)

    def _draw_mem(self, win: curses.window, h: int, width: int, snap: Snapshot) -> None:
        mem = snap.mem
        label_w = 4
        text = f"{htop_mem(mem.used)}/{htop_mem(mem.total)}"
        bar_w = max(4, width - label_w - len(text) - 4)
        if h >= 1:
            add(win, 0, 0, " Mem")
            put_meter(win, 0, label_w + 1, self._mem_bar(mem, bar_w), 0)
            add(win, 0, label_w + bar_w + 3, text)
        if h >= 2:
            swap = f"{htop_mem(mem.swap_used)}/{htop_mem(mem.swap_total)}"
            add(win, 1, 0, " Swp")
            put_meter(win, 1, label_w + 1, self._swap_bar(mem, bar_w), 0)
            add(win, 1, label_w + bar_w + 3, swap)

    def _draw_proc(self, win: curses.window, h: int, width: int, snap: Snapshot) -> None:
        key = {"cpu": lambda p: p.cpu, "mem": lambda p: p.rss, "pid": lambda p: -p.pid}[self.sort]
        procs = sorted(snap.procs, key=key, reverse=True)
        visible = max(0, h - 1)
        if self.proc_offset > max(0, len(procs) - visible):
            self.proc_offset = max(0, len(procs) - visible)
        shown = procs[self.proc_offset : self.proc_offset + visible]
        table = [[
            ("PID", 2), (user_name("USER"), 0), ("%CPU", 2), ("%MEM", 2), ("RSS", 2), ("COMMAND", 1),
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
        lines = layout(table, width - 1, ellipsis=False)
        add(win, 0, 0, lines[0].ljust(width - 1), self.proc_header_attr, width - 1)
        for i, line in enumerate(lines[1:]):
            add(win, 1 + i, 0, line)

    def _help(self, stdscr: curses.window, h: int, w: int) -> None:
        lines = [
            "xtop",
            "",
            "布局写在 xtop/layout.py",
            "一行 = (高度, [(面板, 宽度权重), ...])",
            "高度 0 = 吃掉剩余空间",
            "面板: cpu mem proc",
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
    parser.add_argument("-d", "--delay", type=float, default=1.0, help="host sample interval seconds")
    parser.add_argument("--refresh", type=float, default=3.0, help="screen refresh interval seconds")
    parser.add_argument("-l", "--layout", default=PRESET_ORDER[0], choices=list(PRESETS), help="layout preset")
    parser.add_argument("-v", "--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if args.delay < 0.2:
        parser.error("--delay must be >= 0.2")
    if args.refresh < 0.05:
        parser.error("--refresh must be >= 0.05")
    if not sys.stdout.isatty():
        print("xtop needs a terminal", file=sys.stderr)
        return 1
    os.environ.setdefault("ESCDELAY", "25")
    app = App(args.delay, args.layout, args.refresh)
    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
