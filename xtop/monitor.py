"""Independent host/GPU sampling loops and a non-blocking snapshot interface."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace

from .collect import Collector, CpuStat, MemStat, Snapshot, _mounts
from .gpu import make_gpu_provider


@dataclass(frozen=True)
class PollIntervals:
    system: float = 1.0
    io: float = 1.0
    process: float = 1.0
    mounts: float = 30.0


@dataclass
class _State:
    cpu: CpuStat = field(default_factory=CpuStat)
    mem: MemStat = field(default_factory=MemStat)
    load: tuple[float, float, float] = (0.0, 0.0, 0.0)
    uptime: float = 0.0
    disks: list = field(default_factory=list)
    nets: list = field(default_factory=list)
    mounts: list = field(default_factory=list)
    procs: list = field(default_factory=list)
    nproc: int = 0
    threads: int = 0
    running: int = 0
    gpus: list = field(default_factory=list)
    gpu_error: str = ""
    gpu_attempt: float = 0.0
    gpu_updated: float = 0.0
    errors: dict[str, str] = field(default_factory=dict)
    system_samples: int = 0


class Monitor:
    """Collect without blocking the UI; retain last-good data on failures."""

    def __init__(self, interval: float = 1.0, gpu_provider=None) -> None:
        base = max(0.2, interval)
        self.intervals = PollIntervals(system=base, io=base, process=base)
        self.collector = Collector()
        self.gpu = gpu_provider if gpu_provider is not None else make_gpu_provider()
        self.state = _State()
        self._snapshot: Snapshot | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._gpu_wake = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            return
        self._threads = [
            threading.Thread(target=self._host_loop, name="xtop-host", daemon=True),
            threading.Thread(target=self._gpu_loop, name="xtop-gpu", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._gpu_wake.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
        # Never shut NVML down underneath a driver call that did not return.
        if not any(thread.is_alive() for thread in self._threads):
            try:
                self.gpu.close()
            except Exception:
                pass
        self._threads = []

    def refresh(self) -> None:
        self._wake.set()
        self._gpu_wake.set()

    def snapshot(self) -> Snapshot | None:
        with self._lock:
            snap = self._snapshot
            attempt = self.state.gpu_attempt
            updated = self.state.gpu_updated
        stale_after = max(3.0, self.gpu.metric_interval * 3)
        if snap and attempt > updated and time.monotonic() - attempt > stale_after:
            return replace(snap, gpu_error=snap.gpu_error or "GPU data stale")
        return snap

    @staticmethod
    def _delay(interval: float, failures: int) -> float:
        return min(30.0, interval * (2 ** min(failures, 5)))

    def _publish_locked(self) -> None:
        state = self.state
        self._snapshot = Snapshot(
            host=self.collector.host,
            uptime=state.uptime,
            load=state.load,
            cpu=state.cpu,
            mem=state.mem,
            gpus=state.gpus,
            gpu_error=state.gpu_error,
            disks=state.disks,
            nets=state.nets,
            mounts=state.mounts,
            procs=state.procs,
            nproc=state.nproc,
            threads=state.threads,
            running=state.running,
            cpu_hist=list(self.collector.cpu_hist),
            net_hist=list(self.collector.net_hist),
            disk_hist=list(self.collector.disk_hist),
            ready=state.system_samples >= 2,
            errors=dict(state.errors),
        )

    def _set_gpu_metrics_locked(self, rows: list, retain_missing: bool = False) -> None:
        """Refresh metrics without dropping the slower process-list sample."""
        def identity(gpu):
            return gpu.uuid or f"{gpu.index}:{gpu.name}"

        old = {identity(gpu): gpu for gpu in self.state.gpus}
        merged = [
            replace(gpu, procs=list(old[identity(gpu)].procs)) if identity(gpu) in old else gpu
            for gpu in rows
        ]
        if retain_missing:
            seen = {identity(gpu) for gpu in rows}
            merged.extend(gpu for key, gpu in old.items() if key not in seen)
        self.state.gpus = merged

    def _host_loop(self) -> None:
        intervals = self.intervals
        due = {"system": 0.0, "io": 0.0, "process": 0.0, "mounts": 0.0}
        previous = {"io": None, "process": None}
        failures = {name: 0 for name in due}
        while not self._stop.is_set():
            if self._wake.is_set():
                self._wake.clear()
                due = {name: 0.0 for name in due}
            now = time.monotonic()
            for name in ("system", "mounts", "io", "process"):
                if now < due[name]:
                    continue
                interval = getattr(intervals, name)
                try:
                    if name == "system":
                        cpu = self.collector._cpu()
                        mem = self.collector._mem()
                        load, uptime = self.collector._load()
                        if not mem.total:
                            raise RuntimeError("memory data unavailable")
                        self.collector.cpu_hist.append(cpu.total.busy)
                        with self._lock:
                            self.state.cpu, self.state.mem = cpu, mem
                            self.state.load, self.state.uptime = load, uptime
                            self.state.system_samples += 1
                    elif name == "io":
                        dt = None if previous[name] is None else max(1e-6, now - previous[name])
                        disks = self.collector._disks(dt, refresh_topology=False)
                        nets = self.collector._nets(dt)
                        previous[name] = now
                        self.collector.disk_hist.append(sum(d.read_bps + d.write_bps for d in disks))
                        self.collector.net_hist.append(sum(n.rx_bps + n.tx_bps for n in nets))
                        with self._lock:
                            self.state.disks, self.state.nets = disks, nets
                    elif name == "process":
                        dt = None if previous[name] is None else max(1e-6, now - previous[name])
                        with self._lock:
                            mem_total = self.state.mem.total
                        procs, nproc, threads, running = self.collector._procs(dt, mem_total)
                        previous[name] = now
                        with self._lock:
                            self.state.procs = procs
                            self.state.nproc, self.state.threads = nproc, threads
                            self.state.running = running
                    else:
                        self.collector._refresh_disk_topology()
                        mounts = _mounts()
                        with self._lock:
                            self.state.mounts = mounts
                    failures[name] = 0
                    with self._lock:
                        self.state.errors.pop(name, None)
                        self._publish_locked()
                except Exception as exc:
                    failures[name] += 1
                    with self._lock:
                        self.state.errors[name] = str(exc)
                        self._publish_locked()
                due[name] = now + self._delay(interval, failures[name])
            wait = max(0.02, min(due.values()) - time.monotonic())
            self._stop.wait(min(wait, 0.25))

    def _gpu_loop(self) -> None:
        metric_due = process_due = 0.0
        metric_failures = process_failures = 0
        while not self._stop.is_set():
            if self._gpu_wake.is_set():
                self._gpu_wake.clear()
                metric_due = process_due = 0.0
            now = time.monotonic()
            if now >= metric_due:
                with self._lock:
                    self.state.gpu_attempt = now
                try:
                    rows, error = self.gpu.metrics()
                except Exception as exc:
                    rows, error = [], str(exc)
                if error:
                    metric_failures += 1
                    if rows:
                        with self._lock:
                            self._set_gpu_metrics_locked(rows, retain_missing=True)
                else:
                    metric_failures = 0
                    with self._lock:
                        self._set_gpu_metrics_locked(rows)
                        self.state.gpu_updated = time.monotonic()
                with self._lock:
                    self.state.gpu_error = error
                    self._publish_locked()
                metric_due = now + self._delay(self.gpu.metric_interval, metric_failures)
            if now >= process_due:
                with self._lock:
                    current = self.state.gpus
                try:
                    rows, error = self.gpu.processes(current)
                except Exception as exc:
                    rows, error = current, str(exc)
                if error:
                    process_failures += 1
                else:
                    process_failures = 0
                    with self._lock:
                        self.state.gpus = rows
                with self._lock:
                    if error and not self.state.gpu_error:
                        self.state.gpu_error = error
                    self._publish_locked()
                process_due = now + self._delay(self.gpu.process_interval, process_failures)
            self._stop.wait(0.1)
