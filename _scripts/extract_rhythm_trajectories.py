"""Per-frame rhythm-anchored trajectory extraction for 9 recordings.

3 song-sparrow BCRs × 3 recordings each. Axes:
  X = onset strength (smoothed)
  Y = inter-onset interval rolling mean (rhythm tempo)
  Z = normalized time position [0..1]

Pure rhythm; no tonal features. If birds have rhythmic accents and the
accents are region-mediated, recordings from the same BCR should have
similar (x, y) shapes at similar z, and different BCRs should diverge.

Output:
  acoustic_features/rhythm_trajectories_v1.json  — 9 trajectories ready
  for Blender curve build
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
OUT_JSON = ROOT / "acoustic_features" / "rhythm_trajectories_v1.json"
OUT_AUDIO_DIR = ROOT / "viz" / "audio_for_trajectories"
OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = "song sparrow"
TARGET_BCRS = [
    "COASTAL_CALIFORNIA",
    "SONORAN_AND_MOJAVE_DESERTS",
    "NEW_ENGLAND/MID-ATLANTIC_COAST",
]
N_PER_BCR = 10
TARGET_SR = 22050
HOP_MS = 25          # 40 frames/sec — fine enough for rhythm
HOP_LENGTH = int(TARGET_SR * HOP_MS / 1000)
N_FFT = 1024


def pick_recordings():
    meta = pd.read_parquet(META)
    sub = meta[(meta["common_name"] == SPECIES) & (meta["BCR_NAME"].isin(TARGET_BCRS))].copy()
    sub = sub[sub["q_score"].isin(["A", "B"])]
    # Parse "M:SS" length strings to seconds
    def to_sec(s):
        try:
            if isinstance(s, str) and ":" in s:
                m, ss = s.split(":")
                return int(m) * 60 + int(ss)
            return float(s)
        except: return float("nan")
    sub["length_sec"] = sub["length"].apply(to_sec)
    # Prefer 5-25 sec recordings for clean trajectories
    sub = sub[(sub["length_sec"] >= 4) & (sub["length_sec"] <= 30)]
    selected = []
    for bcr in TARGET_BCRS:
        bcr_sub = sub[sub["BCR_NAME"] == bcr].sort_values("q_score").head(N_PER_BCR * 4)
        # Filter to recordings whose audio actually exists on disk
        ok = []
        for _, r in bcr_sub.iterrows():
            audio_candidates = list(AUDIO_DIR.glob(f"XC{r['id']}.*"))
            if audio_candidates:
                r2 = r.to_dict()
                r2["audio_path"] = str(audio_candidates[0])
                ok.append(r2)
            if len(ok) >= N_PER_BCR: break
        selected.extend(ok)
    return selected


def extract_one(rec):
    """Returns dict with trajectory + metadata."""
    audio_path = rec["audio_path"]
    print(f"  extracting {audio_path}  (id={rec['id']}, bcr={rec['BCR_NAME']})")
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, duration=30.0)
    if len(y) < TARGET_SR * 0.5:
        return None
    # Normalize loudness
    y = y / (np.max(np.abs(y)) + 1e-9)

    # Onset envelope (rhythm energy over time)
    onset_env = librosa.onset.onset_strength(
        y=y, sr=TARGET_SR, hop_length=HOP_LENGTH, n_fft=N_FFT, aggregate=np.median
    )
    # Smooth slightly so we get rhythm-level features, not noise
    onset_env_smooth = librosa.util.normalize(onset_env)

    # Detect onsets for IOI computation
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=TARGET_SR, hop_length=HOP_LENGTH,
        units="time", backtrack=False
    )

    # IOI rolling mean per frame (interpolate onset times into per-frame IOI)
    n_frames = len(onset_env)
    frame_times = np.arange(n_frames) * HOP_LENGTH / TARGET_SR
    # For each frame, find the most recent K=3 onsets and compute mean IOI
    ioi_rolling = np.full(n_frames, np.nan)
    if len(onsets) >= 2:
        iois = np.diff(onsets)  # gaps between consecutive onsets
        # For each frame, find onsets that occurred before it
        for i, ft in enumerate(frame_times):
            past_onsets = onsets[onsets <= ft]
            if len(past_onsets) >= 2:
                # mean of last 3 IOIs
                recent_iois = np.diff(past_onsets[-4:])
                ioi_rolling[i] = float(np.mean(recent_iois))
    # Fill NaN forward then backward
    ioi_series = pd.Series(ioi_rolling).ffill().bfill().fillna(0).values

    # Normalize for viz
    onset_norm = onset_env_smooth  # already in [0, 1]
    ioi_norm = np.clip(ioi_series / (np.median(ioi_series[ioi_series > 0]) + 1e-9), 0, 4) / 4  # 0..1
    time_norm = frame_times / max(frame_times[-1], 1e-6)

    return {
        "id": str(rec["id"]),
        "bcr": rec["BCR_NAME"],
        "lat": float(rec["lat_num"]) if pd.notna(rec.get("lat_num")) else None,
        "lon": float(rec["lon_num"]) if pd.notna(rec.get("lon_num")) else None,
        "ssp": str(rec.get("ssp", "") or ""),
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
    print(f"=== Picking {N_PER_BCR} song-sparrow recordings per BCR ===")
    recs = pick_recordings()
    print(f"  selected {len(recs)} recordings:")
    for r in recs:
        print(f"    XC{r['id']:<8s}  {r['BCR_NAME']:<35s}  len={r['length_sec']:.1f}s  q={r['q_score']}")

    print(f"\n=== Extracting per-frame rhythm trajectories ===")
    trajectories = []
    for r in recs:
        t = extract_one(r)
        if t is not None:
            trajectories.append(t)

    payload = {
        "schema_version": "rhythm_trajectories_v1",
        "species": SPECIES,
        "bcrs": TARGET_BCRS,
        "n_per_bcr": N_PER_BCR,
        "axes": {
            "x": "onset_strength (smoothed, 0-1)",
            "y": "ioi_rolling (inter-onset interval rolling mean, normalized 0-1)",
            "z": "time_norm (0-1 song progression)",
        },
        "hop_ms": HOP_MS,
        "target_sr": TARGET_SR,
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_trajectories": len(trajectories),
        "trajectories": trajectories,
    }
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"\n  -> {OUT_JSON.relative_to(ROOT)}  ({OUT_JSON.stat().st_size:,} bytes)")
    print(f"  total trajectories: {len(trajectories)}")

    # Also copy audio files to viz dir for easy Blender access
    import shutil
    for t in trajectories:
        src = pathlib.Path(t["audio_path"])
        dst = OUT_AUDIO_DIR / f"XC{t['id']}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
    print(f"  audio copied to {OUT_AUDIO_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
