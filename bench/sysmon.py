"""系统资源采样：10Hz 后台线程记录 CPU/RSS/NPU load/温度。

用法：
    mon = Sysmon()
    mon.start()
    ... 被测负载 ...
    series = mon.stop()          # list[dict]，可写 jsonl
    summary = mon.summary()      # 峰值/均值汇总，进结果 JSON

NPU load 走 debugfs /sys/kernel/debug/rknpu/load，默认 root-only。
不可读时该字段为 null 并一次性告警（一次性授权：sudo chmod o+r /sys/kernel/debug/rknpu/load）。
"""
import threading
import time
from pathlib import Path

import psutil

HZ = 10
NPU_LOAD = Path("/sys/kernel/debug/rknpu/load")
ZONES = {  # thermal_zone -> 名（check_env.sh 同款口径）
    "soc": 0, "bigcore": 1, "littlecore": 2, "ddr": 3, "npu": 4, "gpu": 5,
}


def _read_temp(zone: int) -> float | None:
    try:
        return int(Path(f"/sys/class/thermal/thermal_zone{zone}/temp").read_text()) / 1000
    except OSError:
        return None


def _read_npu_load() -> str | None:
    try:
        return NPU_LOAD.read_text().strip()
    except OSError:
        return None


class Sysmon:
    def __init__(self, hz: int = HZ):
        self.interval = 1.0 / hz
        self.proc = psutil.Process()
        self.series: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._npu_warned = False

    def _sample(self) -> dict:
        npu = _read_npu_load()
        if npu is None and not self._npu_warned:
            print("[sysmon] rknpu load 不可读（需 sudo chmod o+r），NPU 占用记 null")
            self._npu_warned = True
        return {
            "ts": round(time.monotonic(), 3),
            "cpu_pct": psutil.cpu_percent(interval=None),  # 自上次调用间隔
            "rss_mb": round(self.proc.memory_info().rss / 1e6, 1),
            "npu_load": npu,
            "temp": {k: _read_temp(z) for k, z in ZONES.items()},
        }

    def _loop(self) -> None:
        psutil.cpu_percent(interval=None)  # 零点
        while not self._stop.is_set():
            self.series.append(self._sample())
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.series.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.series

    def summary(self) -> dict:
        if not self.series:
            return {}
        cpu = [s["cpu_pct"] for s in self.series]
        rss = [s["rss_mb"] for s in self.series]
        out = {
            "samples": len(self.series),
            "cpu_pct_mean": round(sum(cpu) / len(cpu), 1),
            "cpu_pct_peak": max(cpu),
            "rss_mb_peak": max(rss),
        }
        for name in ZONES:
            vals = [s["temp"][name] for s in self.series if s["temp"][name] is not None]
            if vals:
                out[f"temp_{name}_peak_c"] = round(max(vals), 1)
        loads = [s["npu_load"] for s in self.series if s["npu_load"]]
        if loads:
            out["npu_load_last"] = loads[-1]
        return out
