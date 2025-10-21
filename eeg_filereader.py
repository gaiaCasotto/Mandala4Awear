import time
from collections import deque
import numpy as np

AWear_SAMPLE_RATE_HZ = 256.0 

# ---------------- EEG utilities ----------------

# =========================
# EEG streaming + classifier
# =========================
def load_eeg_file(path: str) -> np.ndarray:
    with open(path, "r") as f:
        txt = f.read().replace(",", " ")
    arr = np.fromstring(txt, sep=" ")
    arr = arr[np.isfinite(arr)].astype(np.float32)
    if arr.size:
        arr -= np.mean(arr)
    return arr


class OfflineEEGFeeder:
    def __init__(self, paths, fs=256.0, chunk=32, speed=1.0, loop=True, buffer_s=8.0):
        self.fs=float(fs); self.chunk=int(chunk); self.speed=float(speed)
        self.loop=bool(loop); self.paths=list(paths)
        self.tracks=[load_eeg_file(p) for p in paths]
        if not self.tracks: raise ValueError("No EEG files loaded.")
        self.i=0; self.idx=0; self.dt=self.chunk/(self.fs*self.speed)
        self.buf=deque(maxlen=int(buffer_s*self.fs))
    @property
    def n_tracks(self): return len(self.tracks)
    def set_track(self, i): self.i=int(i)%len(self.tracks); self.idx=0; self.buf.clear()
    def step_once(self):
        x=self.tracks[self.i]
        if self.idx>=len(x):
            if self.loop: self.idx=0
            else: return False
        j=min(self.idx+self.chunk, len(x))
        self.buf.extend(x[self.idx:j].tolist()); self.idx=j; return True
    def get_buffer(self):
        return np.array(self.buf, dtype=np.float32) if self.buf else np.array([], np.float32)
    def sleep_dt(self): time.sleep(self.dt)

def compute_psd(buffer: np.ndarray, fs: float, win_s: float):
    if buffer.size==0: return np.array([0.0]), np.array([0.0])
    N=int(win_s*fs); x=buffer[-N:] if buffer.size>=N else buffer
    if x.size < max(16, int(0.25*N)):
        freqs=np.fft.rfftfreq(max(x.size,2), 1.0/fs)
        return freqs, np.zeros_like(freqs, np.float32)
    w=np.hanning(len(x)); X=np.fft.rfft(x*w)
    psd=(np.abs(X)**2)/(fs*np.sum(w**2))
    freqs=np.fft.rfftfreq(len(x), 1.0/fs)
    return freqs, psd.astype(np.float32)

def bandpower(psd, freqs, fmin, fmax):
    idx=(freqs>=fmin)&(freqs<fmax)
    if not np.any(idx): return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))



# ---------- NEW: simple streaming feeder ----------
class LiveEEGStreamFeeder:
    """
    Thread-safe-ish ring buffer for live, single-channel EEG samples.
    - push(samples): append a list/ndarray of floats
    - get_buffer(): returns a numpy array of the most recent buffer_s seconds
    """
    def __init__(self, fs: float, buffer_s: float = 8.0):
        self.fs = float(fs)
        self.maxlen = int(self.fs * buffer_s)
        self.buf = deque(maxlen=self.maxlen)

    def push(self, samples):
        if samples is None:
            return
        arr = np.asarray(samples, dtype=np.float32).ravel()
        self.buf.extend(arr.tolist())

    def get_buffer(self):
        if not self.buf:
            return np.zeros(self.maxlen, dtype=np.float32)
        return np.asarray(self.buf, dtype=np.float32)

    def step_once(self):
        # kept for API compatibility with your original loop
        pass


class LiveArousalClassifier:
    def __init__(self, 
                 fs=256.0, 
                 lf=(4.0,12.0), 
                 hf=(13.0,40.0),
                 # thresholds between categories (length 3: CALM|MOD|HIGH|EXT)
                 boundaries=(3.0, 5.0, 9.0),
                 hysteresis_frac=0.20,
                 win_s=4.0):
        self.fs = fs; 
        self.lf = lf; 
        self.hf = hf
        self.boundaries = tuple(float(b) for b in boundaries)
        self.hyst = float(hysteresis_frac)
        self.thr_on = self.boundaries
        self.thr_off = tuple(b * (1.0 - self.hyst) for b in self.boundaries)
        self.win_s = float(win_s)
        self.state = "CALM" # calm
        self.last_ratio = np.nan


    def _apply_hysteresis_step(self, ratio: float) -> bool:
        """
        Single-step hysteresis state machine.
        Returns True if state changed, else False.
        """
        prev = self.state

        if self.state == "CALM":
            # CALM -> MODERATE if cross upward on-threshold[0]
            if ratio >= self.thr_on[0]:
                self.state = "MOD-STRESS"

        elif self.state == "MOD-STRESS":
            # MODERATE -> HIGH on-threshold[1]; back to CALM if below off-threshold[0]
            if ratio >= self.thr_on[1]:
                self.state = "HIGH-STRESS"
            elif ratio < self.thr_off[0]:
                self.state = "CALM"

        elif self.state == 2:
            # HIGH -> EXTREME on-threshold[2]; back to MODERATE if below off-threshold[1]
            if ratio >= self.thr_on[2]:
                self.state = "EXTREME-STRESS"
            elif ratio < self.thr_off[1]:
                self.state = "HIGH-STRESS"

        else:  # self.level == 3 (EXTREME)
            # EXTREME -> HIGH if drops below off-threshold[2]
            if ratio < self.thr_off[2]:
                self.state = "HIGH-STRESS"

        return self.state != prev
    
    '''
    keep in mind:
    1) Ratios are unitless and scale-sensitive: If LF power is very small (close to 0), 
    the ratio can blow up because of division.
    -> add 1e-12 in the code — to avoid divide-by-zero.
    '''
    def update(self, buffer):
        '''
        < 3 → "CALM / RELAXED"
        3–5 → "MODERATE STRESS"
        5–9 → "HIGH STRESS"
        > 9 → "EXTREME STRESS"       
        '''
        freqs, psd = compute_psd(buffer, self.fs, self.win_s)
        lf_p = bandpower(psd, freqs, *self.lf)
        hf_p = bandpower(psd, freqs, *self.hf)
        ratio = (hf_p + 1e-12)/(lf_p + 1e-12) # > 3 é stressato
        
        """
        changed = True
        safety = 0
        while changed and safety < 4:
            changed = self._apply_hysteresis_step(ratio)
            safety += 1
        """
        
        if ratio < 3:
            self.state = "CALM" #RELAXED
        elif ratio < 5.0:
            self.state = "MOD-STRESS"
        elif ratio < 8.0:
            self.state = "HIGH-STRESS"
        else:
            self.state = "EXTREME-STRESS"
        
        prev = self.state
        self.last_ratio=ratio

        return self.state, ratio, (self.state!=prev)
    

import pandas as pd

class LiveArousalFromFeatures:
    def __init__(self,
                 boundaries=(3.0, 5.0, 9.0),   # CALM|MOD|HIGH|EXT thresholds on ratio
                 hysteresis_frac=0.20,
                 hf_cols=None,                 # columns considered “high-frequency power” proxies (dB)
                 lf_cols=None):                # columns considered “low-frequency power” proxies (dB)
        self.boundaries = tuple(float(b) for b in boundaries)
        self.hyst = float(hysteresis_frac)
        self.thr_on  = self.boundaries
        self.thr_off = tuple(b * (1.0 - self.hyst) for b in self.boundaries)
        self.state = "CALM"
        self.last_ratio = np.nan

        # reasonable defaults given your schema; we’ll only use the ones present
        self.hf_cols = hf_cols or ["FOCUS_ist_dB", "GLOBAL_FOCUS_max_dB", "TGA_ist_dB", "TGA_avg_dB", "GLOBAL_TGA_max_dB"]
        self.lf_cols = lf_cols or ["TABR_ist_dB", "TABR_avg_dB", "GLOBAL_TABR_min_dB", "GLOBAL_TGA_min_dB", "GLOBAL_FOCUS_min_dB"]

    @staticmethod
    def _db_to_lin(x_db: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return np.power(10.0, x_db.astype(float) / 10.0)

    def _apply_hysteresis_step(self, ratio: float) -> bool:
        prev = self.state

        if self.state == "CALM":
            if ratio >= self.thr_on[0]:
                self.state = "MOD-STRESS"

        elif self.state == "MOD-STRESS":
            if ratio >= self.thr_on[1]:
                self.state = "HIGH-STRESS"
            elif ratio < self.thr_off[0]:
                self.state = "CALM"

        elif self.state == "HIGH-STRESS":
            if ratio >= self.thr_on[2]:
                self.state = "EXTREME-STRESS"
            elif ratio < self.thr_off[1]:
                self.state = "MOD-STRESS"

        else:  # EXTREME-STRESS
            if ratio < self.thr_off[2]:
                self.state = "HIGH-STRESS"

        return self.state != prev

    def _compute_ratio_from_row(self, row: pd.Series) -> float:
        # use whichever columns exist in the row
        hf_present = [c for c in self.hf_cols if c in row.index and pd.notna(row[c])]
        lf_present = [c for c in self.lf_cols if c in row.index and pd.notna(row[c])]
        if not hf_present or not lf_present:
            # fallback: if nothing present, keep last ratio
            return float(self.last_ratio) if np.isfinite(self.last_ratio) else 1.0

        # linear-mean over available columns
        hf_lin = np.mean([10.0**(row[c]/10.0) for c in hf_present])
        lf_lin = np.mean([10.0**(row[c]/10.0) for c in lf_present])
        ratio = (hf_lin + 1e-12) / (lf_lin + 1e-12)
        return float(ratio)

    def update_row(self, row: pd.Series):
        """Update state using a single long_df row (feature snapshot)."""
        prev = self.state
        ratio = self._compute_ratio_from_row(row)

        # Either use hysteresis:
        changed = self._apply_hysteresis_step(ratio)

        # Or use straight thresholds (uncomment to bypass hysteresis):
        # if ratio < self.boundaries[0]:
        #     self.state = "CALM"
        # elif ratio < self.boundaries[1]:
        #     self.state = "MOD-STRESS"
        # elif ratio < self.boundaries[2]:
        #     self.state = "HIGH-STRESS"
        # else:
        #     self.state = "EXTREME-STRESS"
        # changed = (self.state != prev)

        self.last_ratio = ratio
        return self.state, ratio, changed

    def update_batch(self, df: pd.DataFrame):
        """Vectorized pass that returns states/ratios aligned to df.index."""
        states = []
        ratios = []
        for _, row in df.iterrows():
            s, r, _ = self.update_row(row)
            states.append(s); ratios.append(r)
        return pd.DataFrame({"state": states, "ratio": ratios}, index=df.index)

    