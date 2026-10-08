"""Where live data comes from. A source yields one Chunk dict per `chunk_sec` seconds:
  {patient, t, fs, num_dt, ecg, ppg, resp (arrays at fs), spo2, rr, temp (lists every num_dt s), thermal (2-D list or None)}
ReplaySource replays a REAL recording from the dataset cache at real-time speed (a demo without hardware, never synthetic).
Hardware readers implement the same interface; see HardwareSource."""
from __future__ import annotations
import asyncio
import numpy as np
from ..pipeline.windows import load_npz


class SensorSource:
    chunk_sec = 1.0

    async def chunks(self):  # async generator of Chunk dicts
        raise NotImplementedError
        yield


class ReplaySource(SensorSource):
    def __init__(self, npz_path, speed=1.0, loop=False):
        self.c, self.speed, self.loop = load_npz(npz_path), speed, loop

    async def chunks(self):
        c, fs, dt = self.c, self.c["fs"], self.c["num_dt"]
        dur = int(max(len(c[k]) / fs for k in ("ecg", "ppg", "resp") if k in c) if any(k in c for k in ("ecg", "ppg", "resp"))
                  else max(len(c[k]) * dt for k in ("spo2", "hr", "rr", "temp") if k in c))
        while True:
            for s in range(dur):
                ch = dict(patient=c["patient"], t=float(s + 1), fs=fs, num_dt=dt, thermal=None)
                for k in ("ecg", "ppg", "resp"):
                    ch[k] = c[k][s * fs:(s + 1) * fs].tolist() if k in c else None
                for k in ("spo2", "rr", "temp"):
                    ch[k] = c[k][int(s / dt):int((s + 1) / dt)].tolist() if k in c else None
                yield ch
                await asyncio.sleep(1.0 / self.speed)
            if not self.loop:
                return


class HardwareSource(SensorSource):
    """Implement for your devices: poll each driver, assemble one Chunk per second with the same keys as above
    (waveforms resampled to cfg['fs'] Hz; SpO2/RR/temperature as one value per num_dt; thermal as a 2-D list in deg C),
    and `yield` it. Nothing in the model code changes."""
