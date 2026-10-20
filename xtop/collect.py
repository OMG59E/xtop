"""Read host counters from procfs and nvidia-smi. Stdlib only."""

from __future__ import annotations

import csv
import os
import pwd
import re
import shutil
import socket
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
CLK_TCK = os.sysconf("SC_CLK_TCK")
HIST = 60

_PART = re.compile(
    r"^(nvme\d+n\d+p\d+|mmcblk\d+p\d+|sd[a-z]+\d+|vd[a-z]+\d+|hd[a-z]+\d+|xvd[a-z]+\d+)$"
)
_SKIP_BLK = re.compile(r"^(loop|ram|fd|sr|zram)")
_SKIP_NET = re.compile(r"^(lo|veth|docker|br-|virbr|tun|tap|cni|flannel)")


@dataclass
class CpuMeter:
    """Percent of one interval. Order matches htop's default CPU bar."""

    user: float = 0.0
    nice: float = 0.0
    system: float = 0.0
    irq: float = 0.0
    softirq: float = 0.0
    steal: float = 0.0
    guest: float = 0.0
    iowait: float = 0.0

    @property
    def busy(self) -> float:
        return self.user + self.nice + self.system + self.irq + self.softirq + self.steal + self.guest


@dataclass
class CpuStat:
    total: CpuMeter = field(default_factory=CpuMeter)
    cores: list[CpuMeter] = field(default_factory=list)
    freq_mhz: list[float] = field(default_factory=list)
    model: str = ""


@dataclass
class MemStat:
    """Bytes. used is htop's green: total - free - buffers - cache."""

    total: int = 0
    used: int = 0
    buffers: int = 0
    cache: int = 0
    shared: int = 0
    available: int = 0
    swap_total: int = 0
    swap_used: int = 0
    swap_cache: int = 0


@dataclass
class GpuProc:
    gpu: int
    pid: int
    user: str
    mem_mib: float | None
    cmd: str


@dataclass
class GpuStat:
    index: int
    name: str
    util: float | None
    mem_util: float | None
    mem_used: float | None
    mem_total: float | None
    temp: float | None
    power: float | None
    power_limit: float | None
    procs: list[GpuProc] = field(default_factory=list)


@dataclass
class DiskStat:
    name: str
    read_bps: float
    write_bps: float
    util: float
    mounts: str = ""
    total: int = 0
    free: int = 0


@dataclass
class NetStat:
    name: str
    rx_bps: float
    tx_bps: float


@dataclass
class FsStat:
    mount: str
    fstype: str
    total: int
    used: int


@dataclass
class ProcStat:
    pid: int
    user: str
    state: str
    pri: int
    nice: int
    cpu: float
    mem: float
    virt: int
    rss: int
    comm: str
    cmd: str


@dataclass
class Snapshot:
    host: str
    uptime: float
    load: tuple[float, float, float]
    cpu: CpuStat
    mem: MemStat
    gpus: list[GpuStat]
    gpu_error: str
    disks: list[DiskStat]
    nets: list[NetStat]
    mounts: list[FsStat]
    procs: list[ProcStat]
    nproc: int
    threads: int
    running: int
    cpu_hist: list[float]
    net_hist: list[float]
    disk_hist: list[float]
    ready: bool


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _pct(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * part / total


def _cpu_meter(nums: list[int], prev: list[int] | None) -> CpuMeter:
    """/proc/stat: user nice system idle iowait irq softirq steal guest guest_nice."""
    if prev is None or len(prev) != len(nums):
        return CpuMeter()
    delta = [c - p for c, p in zip(nums, prev)]
    if any(v < 0 for v in delta):
        return CpuMeter()
    while len(delta) < 10:
        delta.append(0)
    # guest and guest_nice are already inside user and nice
    total = sum(delta[:8])
    if total <= 0:
        return CpuMeter()

    def part(value: int) -> float:
        return 100.0 * value / total

    return CpuMeter(
        user=part(max(0, delta[0] - delta[8])),
        nice=part(max(0, delta[1] - delta[9])),
        system=part(delta[2]),
        iowait=part(delta[4]),
        irq=part(delta[5]),
        softirq=part(delta[6]),
        steal=part(delta[7]),
        guest=part(delta[8] + delta[9]),
    )


def _parse_stat_line(line: str) -> tuple[str, str, int, int, int, int, int] | None:
    rpar = line.rfind(")")
    lpar = line.find("(")
    if rpar < 0 or lpar < 0:
        return None
    rest = line[rpar + 2 :].split()
    if len(rest) < 22:
        return None
    comm = line[lpar + 1 : rpar]
    try:
        return (
            comm,
            rest[0],
            int(rest[11]),
            int(rest[12]),
            int(rest[15]),
            int(rest[16]),
            int(rest[20]),
        )
    except ValueError:
        return None


def _num(text: str) -> float | None:
    text = text.strip()
    if not text or text.upper() in {"[N/A]", "N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt_model() -> str:
    try:
        for line in _read("/proc/cpuinfo").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _freqs() -> list[float]:
    """Current MHz of every online CPU, index-aligned with /proc/stat cores."""
    base = "/sys/devices/system/cpu"
    try:
        names = os.listdir(base)
    except OSError:
        return []
    idxs = sorted(int(name[3:]) for name in names if re.fullmatch(r"cpu\d+", name))
    vals: list[float] = []
    for idx in idxs:
        path = os.path.join(base, f"cpu{idx}", "cpufreq", "scaling_cur_freq")
        try:
            vals.append(int(_read(path).strip()) / 1000.0)
        except (OSError, ValueError):
            vals.append(0.0)
    return vals


def _block_devices() -> list[str]:
    """Whole physical disks only: a device node backed by real hardware."""
    try:
        names = os.listdir("/sys/block")
    except OSError:
        return []
    out = []
    for name in names:
        if _PART.match(name) or _SKIP_BLK.match(name) or name.startswith("dm-"):
            continue
        if not os.path.exists(f"/sys/block/{name}/device"):
            continue
        out.append(name)
    return out


def _disk_mounts() -> dict[str, list[str]]:
    """Physical disk -> real mountpoints, including those on its partitions and LVM."""
    parent = _dev_parent()
    dm_names = _dm_names()
    mounts: dict[str, list[str]] = {}
    try:
        lines = _read("/proc/mounts").splitlines()
    except OSError:
        return mounts
    for line in lines:
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("/dev/"):
            continue
        mount = parts[1]
        if mount.startswith(("/proc", "/sys", "/dev", "/run", "/snap")):
            continue
        dev = _mount_dev(parts[0], dm_names)
        disk = parent.get(dev)
        if not disk:
            continue
        mounts.setdefault(disk, [])
        if mount not in mounts[disk]:
            mounts[disk].append(mount)
    for name in mounts:
        mounts[name] = _real_mounts(mounts[name])
    return mounts


def _dev_parent() -> dict[str, str]:
    """Map a disk, partition or device-mapper name back to its physical disk."""
    parent: dict[str, str] = {}
    for disk in _block_devices():
        parent[disk] = disk
        base = f"/sys/block/{disk}"
        try:
            for entry in os.listdir(base):
                if os.path.isdir(f"{base}/{entry}") and entry.startswith(disk):
                    parent[entry] = disk
                    for holder in _holders(f"{base}/{entry}/holders"):
                        parent[holder] = disk
                        _dm_chain(holder, disk, parent)
        except OSError:
            continue
        for holder in _holders(f"{base}/holders"):
            parent[holder] = disk
            _dm_chain(holder, disk, parent)
    return parent


def _dm_names() -> dict[str, str]:
    """dm-N -> mapper name, so mounts whose /dev/mapper node is hidden still resolve."""
    names = {}
    try:
        for entry in os.listdir("/sys/block"):
            path = f"/sys/block/{entry}/dm/name"
            if os.path.exists(path):
                names[entry] = _read(path).strip()
    except OSError:
        pass
    return names


def _holders(path: str) -> list[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []


def _disk_usage(mounts: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
    """Sum filesystem size and free bytes of each disk's real mountpoints."""
    usage: dict[str, tuple[int, int]] = {}
    for disk, points in mounts.items():
        total = free = 0
        for mount in points:
            try:
                st = os.statvfs(mount)
            except OSError:
                continue
            total += st.f_frsize * st.f_blocks
            free += st.f_frsize * st.f_bavail
        usage[disk] = (total, free)
    return usage


def _real_mounts(candidates: list[str]) -> list[str]:
    """Keep directory mounts, dropping files and mounts nested under one already kept."""
    dirs = []
    for mount in candidates:
        try:
            if os.path.isdir(mount):
                dirs.append(mount)
        except OSError:
            continue
    kept = []
    for mount in sorted(dirs, key=len):
        if any(mount.startswith(base.rstrip("/") + "/") for base in kept):
            continue
        kept.append(mount)
    kept.sort(key=lambda m: (m != "/", len(m)))
    return kept


def _mount_dev(source: str, dm_names: dict[str, str]) -> str:
    if source.startswith("/dev/mapper/"):
        wanted = source[len("/dev/mapper/") :]
        for dm, name in dm_names.items():
            if name == wanted:
                return dm
        return ""
    return os.path.basename(os.path.realpath(source)) if source.startswith("/dev/") else ""
    try:
        return os.listdir(path)
    except OSError:
        return []


def _dm_chain(dm: str, disk: str, parent: dict[str, str]) -> None:
    """Follow stacked device-mapper (LVM on LVM) back to the same disk."""
    for holder in _holders(f"/sys/block/{dm}/holders"):
        if parent.get(holder) == disk:
            continue
        parent[holder] = disk
        _dm_chain(holder, disk, parent)
    """name -> (read_bytes, write_bytes, io_ms). Sectors are 512 bytes."""
    wanted = set(_block_devices())
    out: dict[str, tuple[int, int, int]] = {}
    try:
        lines = _read("/proc/diskstats").splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) < 14:
            continue
        name = parts[2]
        if name not in wanted:
            continue
        try:
            read_b = int(parts[5]) * 512
            write_b = int(parts[9]) * 512
            io_ms = int(parts[12])
        except ValueError:
            continue
        out[name] = (read_b, write_b, io_ms)
    return out


def _ipv4_ifaces() -> set[str]:
    """Non-loopback interfaces that have an IPv4 address. SIOCGIFCONF, stdlib only."""
    import ctypes
    import ctypes.util
    import socket
    import struct

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    buf = ctypes.create_string_buffer(8192)
    ifconf = struct.pack("iP", ctypes.sizeof(buf), ctypes.addressof(buf))
    arr = ctypes.create_string_buffer(ifconf)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if libc.ioctl(sock.fileno(), 0x8912, arr) < 0:  # SIOCGIFCONF
            return set()
        length = struct.unpack_from("i", arr.raw)[0]
        found: set[str] = set()
        offset = 0
        while offset + 40 <= length:
            name = buf.raw[offset : offset + 16].split(b"\x00", 1)[0].decode()
            family = struct.unpack_from("H", buf.raw, offset + 16)[0]
            if family == socket.AF_INET:
                ip = socket.inet_ntoa(buf.raw[offset + 20 : offset + 24])
                if not ip.startswith("127."):
                    found.add(name)
            offset += 40
        return found
    finally:
        sock.close()


def _diskstats() -> dict[str, tuple[int, int, int]]:
    """name -> (read_bytes, write_bytes, io_ms). Sectors are 512 bytes."""
    wanted = set(_block_devices())
    out: dict[str, tuple[int, int, int]] = {}
    try:
        lines = _read("/proc/diskstats").splitlines()
    except OSError:
        return out
    for line in lines:
        parts = line.split()
        if len(parts) < 14:
            continue
        name = parts[2]
        if name not in wanted:
            continue
        try:
            read_b = int(parts[5]) * 512
            write_b = int(parts[9]) * 512
            io_ms = int(parts[12])
        except ValueError:
            continue
        out[name] = (read_b, write_b, io_ms)
    return out


def _netdev() -> dict[str, tuple[int, int]]:
    wanted = _ipv4_ifaces()
    out: dict[str, tuple[int, int]] = {}
    try:
        lines = _read("/proc/net/dev").splitlines()[2:]
    except OSError:
        return out
    for line in lines:
        if ":" not in line:
            continue
        name, rest = line.split(":", 1)
        name = name.strip()
        if name not in wanted or _SKIP_NET.match(name):
            continue
        cols = rest.split()
        if len(cols) < 9:
            continue
        try:
            out[name] = (int(cols[0]), int(cols[8]))
        except ValueError:
            continue
    return out


_SKIP_FS = {
    "proc", "sysfs", "devtmpfs", "devpts", "cgroup", "cgroup2", "pstore",
    "bpf", "tracefs", "debugfs", "securityfs", "fusectl", "configfs",
    "rpc_pipefs", "autofs", "mqueue", "hugetlbfs", "binfmt_misc", "squashfs",
    "nsfs", "ramfs", "overlay",
}


def _mounts() -> list[FsStat]:
    seen_dev: set[tuple[int, int]] = set()
    rows: list[FsStat] = []
    try:
        lines = _read("/proc/mounts").splitlines()
    except OSError:
        return rows
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount, fstype = parts[1], parts[2]
        if fstype in _SKIP_FS:
            continue
        if mount.startswith(("/proc", "/sys", "/dev", "/run", "/snap")):
            continue
        try:
            st = os.statvfs(mount)
            dev = os.stat(mount)
        except OSError:
            continue
        total = st.f_frsize * st.f_blocks
        if total <= 0:
            continue
        ident = (dev.st_dev, total)
        if ident in seen_dev:
            continue
        seen_dev.add(ident)
        avail = st.f_frsize * st.f_bavail
        rows.append(FsStat(mount, fstype, total, total - avail))
    rows.sort(key=lambda r: (r.mount != "/", -r.total))
    return rows


def _gpu_user(pid: int) -> str:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (OSError, KeyError):
        return "?"


def _gpu_cmd(pid: int, fallback: str) -> str:
    try:
        cmd = _read(f"/proc/{pid}/cmdline").replace("\x00", " ").strip()
    except OSError:
        cmd = ""
    return cmd or fallback or f"[{pid}]"


def _gpus() -> tuple[list[GpuStat], str]:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return [], ""
    query = [
        binary,
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(query, capture_output=True, text=True, timeout=2.0, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "nvidia-smi failed").strip().splitlines()
        return [], err[0] if err else "nvidia-smi failed"
    gpus: dict[int, GpuStat] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        cols = [c.strip() for c in next(csv.reader([line]))]
        if len(cols) < 9:
            continue
        try:
            index = int(cols[0])
        except ValueError:
            continue
        gpus[index] = GpuStat(
            index=index,
            name=cols[1],
            util=_num(cols[2]),
            mem_util=_num(cols[3]),
            mem_used=_num(cols[4]),
            mem_total=_num(cols[5]),
            temp=_num(cols[6]),
            power=_num(cols[7]),
            power_limit=_num(cols[8]),
        )
    apps = [
        binary,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    uuid_of: dict[str, int] = {}
    try:
        listed = subprocess.run(
            [binary, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
        for line in listed.stdout.splitlines():
            cols = [c.strip() for c in next(csv.reader([line]))]
            if len(cols) >= 2:
                uuid_of[cols[1]] = int(cols[0])
        running = subprocess.run(apps, capture_output=True, text=True, timeout=2.0, check=False)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return list(gpus.values()), ""
    if running.returncode == 0:
        for line in running.stdout.splitlines():
            if not line.strip():
                continue
            cols = [c.strip() for c in next(csv.reader([line]))]
            if len(cols) < 4:
                continue
            index = uuid_of.get(cols[0])
            if index is None or index not in gpus:
                continue
            try:
                pid = int(cols[1])
            except ValueError:
                continue
            gpus[index].procs.append(
                GpuProc(index, pid, _gpu_user(pid), _num(cols[3]), _gpu_cmd(pid, cols[2]))
            )
    return [gpus[i] for i in sorted(gpus)], ""


class Collector:
    def __init__(self) -> None:
        self.host = socket.gethostname()
        self.model = _fmt_model()
        self.prev_ts: float | None = None
        self.prev_cpu: list[int] | None = None
        self.prev_cores: list[list[int]] | None = None
        self.prev_disk: dict[str, tuple[int, int, int]] = {}
        self.prev_net: dict[str, tuple[int, int]] = {}
        self.prev_proc: dict[int, tuple[int, float]] = {}
        self.users: dict[int, str] = {}
        self.cpu_hist: deque[float] = deque(maxlen=HIST)
        self.net_hist: deque[float] = deque(maxlen=HIST)
        self.disk_hist: deque[float] = deque(maxlen=HIST)

    def sample(self) -> Snapshot:
        now = time.monotonic()
        dt = None if self.prev_ts is None else max(1e-6, now - self.prev_ts)
        cpu = self._cpu()
        mem = self._mem()
        disks = self._disks(dt)
        nets = self._nets(dt)
        procs, nproc, threads, running = self._procs(dt, mem.total)
        gpus, gpu_error = _gpus()
        load, uptime = self._load()
        self.prev_ts = now

        self.cpu_hist.append(cpu.total.busy)
        self.disk_hist.append(sum(d.read_bps + d.write_bps for d in disks))
        self.net_hist.append(sum(n.rx_bps + n.tx_bps for n in nets))
        cpu.model = self.model

        return Snapshot(
            host=self.host,
            uptime=uptime,
            load=load,
            cpu=cpu,
            mem=mem,
            gpus=gpus,
            gpu_error=gpu_error,
            disks=disks,
            nets=nets,
            mounts=_mounts(),
            procs=procs,
            nproc=nproc,
            threads=threads,
            running=running,
            cpu_hist=list(self.cpu_hist),
            net_hist=list(self.net_hist),
            disk_hist=list(self.disk_hist),
            ready=dt is not None,
        )

    def _load(self) -> tuple[tuple[float, float, float], float]:
        try:
            parts = _read("/proc/loadavg").split()
            load = (float(parts[0]), float(parts[1]), float(parts[2]))
        except (OSError, ValueError, IndexError):
            load = (0.0, 0.0, 0.0)
        try:
            uptime = float(_read("/proc/uptime").split()[0])
        except (OSError, ValueError, IndexError):
            uptime = 0.0
        return load, uptime

    def _cpu(self) -> CpuStat:
        total: list[int] | None = None
        cores: list[list[int]] = []
        try:
            for line in _read("/proc/stat").splitlines():
                if not line.startswith("cpu"):
                    break
                parts = line.split()
                nums = [int(x) for x in parts[1:]]
                while len(nums) < 10:
                    nums.append(0)
                if parts[0] == "cpu":
                    total = nums
                else:
                    cores.append(nums)
        except (OSError, ValueError):
            return CpuStat()
        if total is None:
            return CpuStat()
        meter = _cpu_meter(total, self.prev_cpu)
        if self.prev_cores and len(self.prev_cores) == len(cores):
            core_meters = [_cpu_meter(cur, prev) for cur, prev in zip(cores, self.prev_cores)]
        else:
            core_meters = [CpuMeter() for _ in cores]
        self.prev_cpu = total
        self.prev_cores = cores
        return CpuStat(total=meter, cores=core_meters, freq_mhz=_freqs(), model=self.model)

    def _mem(self) -> MemStat:
        info: dict[str, int] = {}
        try:
            for line in _read("/proc/meminfo").splitlines():
                key, val = line.split(":", 1)
                info[key] = int(val.strip().split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            return MemStat()
        total = info.get("MemTotal", 0)
        free = info.get("MemFree", 0)
        buffers = info.get("Buffers", 0)
        cache = info.get("Cached", 0) + info.get("SReclaimable", 0)
        shared = info.get("Shmem", 0)
        available = info.get("MemAvailable", free)
        swap_total = info.get("SwapTotal", 0)
        swap_free = info.get("SwapFree", 0)
        swap_cache = info.get("SwapCached", 0)
        # htop: used excludes reclaimable cache; shared is drawn out of that used slice
        used = max(0, total - free - buffers - cache)
        return MemStat(
            total=total,
            used=used,
            buffers=buffers,
            cache=max(0, cache - shared),
            shared=shared,
            available=available,
            swap_total=swap_total,
            swap_used=max(0, swap_total - swap_free - swap_cache),
            swap_cache=swap_cache,
        )

    def _disks(self, dt: float | None) -> list[DiskStat]:
        cur = _diskstats()
        rows: list[DiskStat] = []
        mounts = _disk_mounts()
        usage = _disk_usage(mounts)
        if dt is not None:
            for name, (rb, wb, io) in cur.items():
                prev = self.prev_disk.get(name)
                if prev is None:
                    continue
                rows.append(
                    DiskStat(
                        name=name,
                        read_bps=max(0.0, (rb - prev[0]) / dt),
                        write_bps=max(0.0, (wb - prev[1]) / dt),
                        util=min(100.0, max(0.0, (io - prev[2]) / (dt * 10.0))),
                        mounts=",".join(mounts.get(name, [])),
                        total=usage.get(name, (0, 0))[0],
                        free=usage.get(name, (0, 0))[1],
                    )
                )
        rows = [d for d in rows if d.mounts]
        self.prev_disk = cur
        rows.sort(key=lambda d: _block_devices().index(d.name) if d.name in _block_devices() else 0)
        return rows

    def _nets(self, dt: float | None) -> list[NetStat]:
        cur = _netdev()
        rows: list[NetStat] = []
        if dt is not None:
            for name, (rx, tx) in cur.items():
                prev = self.prev_net.get(name)
                if prev is None:
                    continue
                rows.append(
                    NetStat(
                        name=name,
                        rx_bps=max(0.0, (rx - prev[0]) / dt),
                        tx_bps=max(0.0, (tx - prev[1]) / dt),
                    )
                )
        self.prev_net = cur
        rows.sort(key=lambda n: n.rx_bps + n.tx_bps, reverse=True)
        return rows

    def _user(self, uid: int) -> str:
        cached = self.users.get(uid)
        if cached is not None:
            return cached
        try:
            name = pwd.getpwuid(uid).pw_name
        except KeyError:
            name = str(uid)
        self.users[uid] = name
        return name

    def _procs(self, dt: float | None, mem_total: int) -> tuple[list[ProcStat], int, int, int]:
        """Every process visible in /proc, all uids including root. No filter."""
        now = time.monotonic()
        found: dict[int, tuple[int, float]] = {}
        rows: list[ProcStat] = []
        threads = 0
        running = 0
        try:
            names = os.listdir("/proc")
        except OSError:
            return [], 0, 0, 0
        for name in names:
            if not name.isdigit():
                continue
            pid = int(name)
            base = f"/proc/{pid}"
            try:
                parsed = _parse_stat_line(_read(f"{base}/stat"))
                if parsed is None:
                    continue
                comm, state, utime, stime, pri, nice, virt = parsed
                ticks = utime + stime
                rss = int(_read(f"{base}/statm").split()[1]) * PAGE_SIZE
                uid = os.stat(base).st_uid
                nthreads = sum(1 for entry in os.listdir(f"{base}/task") if entry.isdigit())
                cmd = _read(f"{base}/cmdline").replace("\x00", " ").strip() or f"[{comm}]"
            except (OSError, IndexError, ValueError):
                continue
            found[pid] = (ticks, now)
            prev = self.prev_proc.get(pid)
            cpu = 0.0
            if dt and prev is not None:
                cpu = max(0.0, (ticks - prev[0]) / CLK_TCK / dt * 100.0)
            mem = (100.0 * rss / mem_total) if mem_total else 0.0
            threads += nthreads
            if state == "R":
                running += 1
            rows.append(
                ProcStat(pid, self._user(uid), state, pri, nice, cpu, mem, virt, rss, comm, cmd)
            )
        self.prev_proc = found
        return rows, len(rows), threads, running
