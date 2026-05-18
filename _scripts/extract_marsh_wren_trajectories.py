"""Marsh wren rhythm trajectories: 5 per BCR x 3 BCRs = 15.
Same axes (onset_strength, ioi_rolling) as song-sparrow extraction
so cross-species DTW is comparable.
"""
import pathlib, sys, io, json, time
import warnings; warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass
import numpy as np
import pandas as pd
import librosa

ROOT = pathlib.Path(r"D:/Bird Song")
META = ROOT / "range_data" / "xc_metadata_joined_bcr.parquet"
AUDIO_DIR = ROOT / "recordings" / "xeno_canto"
OUT_JSON = ROOT / "acoustic_features" / "rhythm_trajectories_marsh_wren_v1.json"

SPECIES = "marsh wren"
TARGET_BCRS = ["COASTAL_CALIFORNIA", "SONORAN_AND_MOJAVE_DESERTS", "NEW_ENGLAND/MID-ATLANTIC_COAST"]
N_PER_BCR = 10
TARGET_SR = 22050
HOP_MS = 25
HOP_LENGTH = int(TARGET_SR * HOP_MS / 1000)
N_FFT = 1024


def to_sec(s):
    try:
        if isinstance(s, str) and ":" in s:
            m, ss = s.split(":")
            return int(m) * 60 + int(ss)
        return float(s)
    except: return float("nan")


def pick_recordings():
    meta = pd.read_parquet(META)
    sub = meta[(meta["common_name"] == SPECIES) & (meta["BCR_NAME"].isin(TARGET_BCRS))].copy()
    sub = sub[sub["q_score"].isin(["A", "B"])]
    sub["length_sec"] = sub["length"].apply(to_sec)
    sub = sub[(sub["length_sec"] >= 3) & (sub["length_sec"] <= 60)]
    selected = []
    for bcr in TARGET_BCRS:
        bcr_sub = sub[sub["BCR_NAME"] == bcr].sort_values("q_score").head(N_PER_BCR * 4)
        ok = []
        for _, r in bcr_sub.iterrows():
            cands = list(AUDIO_DIR.glob(f"XC{r['id']}.*"))
            if cands:
                r2 = r.to_dict()
                r2["audio_path"] = str(cands[0])
                ok.append(r2)
            if len(ok) >= N_PER_BCR: break
        selected.extend(ok)
    return selected


def extract_one(rec):
    audio_path = rec["audio_path"]
    print(f"  XC{rec['id']}  {rec['BCR_NAME']:<35s} len={rec['length_sec']}s")
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, duration=30.0)
    if len(y) < TARGET_SR * 0.5:
        return None
    y = y / (np.max(np.abs(y)) + 1e-9)
    onset_env = librosa.onset.onset_strength(y=y, sr=TARGET_SR, hop_length=HOP_LENGTH,
                                              n_fft=N_FFT, aggregate=np.median)
    onset_norm = librosa.util.normalize(onset_env)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=TARGET_SR,
                                         hop_length=HOP_LENGTH, units="time", backtrack=False)
    n_frames = len(onset_env)
    frame_times = np.arange(n_frames) * HOP_LENGTH / TARGET_SR
    ioi_rolling = np.full(n_frames, np.nan)
    if len(onsets) >= 2:
        for i, ft in enumerate(frame_times):
            past = onsets[onsets <= ft]
            if len(past) >= 2:
                ioi_rolling[i] = float(np.mean(np.diff(past[-4:])))
    ioi_series = pd.Series(ioi_rolling).ffill().bfill().fillna(0).values
    ioi_med = np.median(ioi_series[ioi_series > 0]) if (ioi_series > 0).any() else 1.0
    ioi_norm = np.clip(ioi_series / (ioi_med + 1e-9), 0, 4) / 4
    time_norm = frame_times / max(frame_times[-1], 1e-6)
    return {
        "id": str(rec["id"]),
        "species": SPECIES,
        "bcr": rec["BCR_NAME"],
        "lat": float(rec["lat_num"]) if pd.notna(rec.get("lat_num")) else None,
        "lon": float(rec["lon_num"]) if pd.notna(rec.get("lon_num")) else None,
        "duration_sec": float(len(y) / TARGET_SR),
        "n_onsets": int(len(onsets)),
        "audio_path": rec["audio_path"],
        "n_frames": int(n_frames),
        "trajectory": {
            "onset_strength": onset_norm.tolist(),
            "ioi_rolling": ioi_norm.tolist(),
            "time_norm": time_norm.tolist(),
        },
    }


def main():
    recs = pick_recordings()
    print(f"=== {len(recs)} marsh wren recordings selected ===")
    trajectories = []
    for r in recs:
        t = extract_one(r)
        if t is not None: trajectories.append(t)
    payload = {
        "schema_version": "rhythm_trajectories_v1_marsh_wren",
        "species": SPECIES,
        "bcrs": TARGET_BCRS,
        "n_per_bcr": N_PER_BCR,
        "axes": {
            "x": "onset_strength (smoothed, 0-1)",
            "y": "ioi_rolling (normalized 0-1)",
            "z": "time_norm (0-1)",
        },
        "n_trajectories": len(trajectories),
        "trajectories": trajectories,
    }
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"-> {OUT_JSON.relative_to(ROOT)}  ({OUT_JSON.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
