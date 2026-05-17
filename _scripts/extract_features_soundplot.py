"""Phase 1, step 2: SoundPlot batch feature extraction.

Loads each fetched audio file, runs SoundPlot's FeatureExtractor pipeline,
saves per-recording acoustic features as one parquet file. Multiprocess
across cores. Resumable (skips already-extracted IDs).

Input:  recordings/xeno_canto/XC*.{mp3,wav,flac,ogg}  (1,344 files)
Output: acoustic_features/soundplot_features_v1.parquet (per-recording)
"""
import sys, pathlib, time, os, warnings, traceback
import multiprocessing as mp
warnings.filterwarnings("ignore")

ROOT = pathlib.Path(r"D:/Bird Song")
SOUNDPLOT = ROOT / "_external" / "SoundPlot"
sys.path.insert(0, str(SOUNDPLOT))
AUDIO_DIR = ROOT / "recordings" / "xeno_canto"
OUT_PARQUET = ROOT / "acoustic_features" / "soundplot_features_v1.parquet"
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)


def extract_one(audio_path_str):
    """Worker: extract SoundPlot features for one recording."""
    import warnings; warnings.filterwarnings("ignore")
    from src.audio import AudioLoader, AudioPreprocessor
    from src.features import FeatureExtractor
    import numpy as np
    audio_path = pathlib.Path(audio_path_str)
    xc_id = audio_path.stem.replace("XC", "")
    t0 = time.time()
    try:
        loader = AudioLoader(target_sr=22050)
        audio, sr = loader.load(str(audio_path))
        # Cap at 60 sec to bound feature-extraction time per recording
        if len(audio) > 60 * sr:
            audio = audio[: 60 * sr]
        pre = AudioPreprocessor(sample_rate=sr)
        audio = pre.preprocess_full(audio, normalize=True, denoise=False)
        if len(audio) < sr * 0.5:  # require ≥0.5 sec
            return {"id": xc_id, "ok": False, "error": "too_short",
                    "duration_sec": len(audio)/sr, "wall_sec": time.time()-t0}
        fx = FeatureExtractor(sample_rate=sr)
        feats = fx.extract_all(audio)
        rec = {"id": xc_id, "ok": True, "error": "",
               "duration_sec": float(len(audio) / sr),
               "wall_sec": time.time() - t0}
        # Flatten feature dict (all scalar values)
        for k, v in feats.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                rec[k] = float(v)
            elif isinstance(v, np.ndarray):
                # Save aggregates for vector features
                rec[f"{k}_mean"] = float(np.nanmean(v))
                rec[f"{k}_std"]  = float(np.nanstd(v))
                rec[f"{k}_min"]  = float(np.nanmin(v))
                rec[f"{k}_max"]  = float(np.nanmax(v))
        return rec
    except Exception as e:
        return {"id": xc_id, "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "duration_sec": None, "wall_sec": time.time()-t0}


def main():
    import pandas as pd
    audio_files = sorted([p for p in AUDIO_DIR.glob("XC*.*")
                          if p.suffix.lower() in (".mp3",".wav",".flac",".ogg",".m4a")])
    print(f"audio files found: {len(audio_files)}")

    # Resume: drop IDs already in the parquet
    if OUT_PARQUET.exists():
        existing = pd.read_parquet(OUT_PARQUET)
        done_ids = set(existing["id"].astype(str))
        print(f"already extracted: {len(done_ids)}")
        to_do = [p for p in audio_files if p.stem.replace("XC","") not in done_ids]
    else:
        existing = pd.DataFrame()
        to_do = audio_files
    print(f"to extract this run: {len(to_do)}")
    if not to_do:
        print("Nothing to do. Exiting.")
        return

    # Worker count: 9950X3D has 16C/32T, leave headroom
    n_workers = min(12, mp.cpu_count() - 2)
    print(f"workers: {n_workers}")
    t_start = time.time()
    done = 0
    err = 0
    new_rows = []
    BATCH_SAVE = 100

    with mp.Pool(n_workers) as pool:
        for rec in pool.imap_unordered(extract_one, [str(p) for p in to_do], chunksize=4):
            new_rows.append(rec)
            done += 1
            if not rec["ok"]:
                err += 1
            if done % 50 == 0 or done == len(to_do):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(to_do) - done) / rate / 60 if rate > 0 else 0
                ok_count = done - err
                print(f"[{done:>5}/{len(to_do)}]  ok={ok_count}  err={err}  "
                      f"elapsed={elapsed/60:.1f}m  eta={eta:.1f}m  rate={rate:.1f}/s")
            if len(new_rows) >= BATCH_SAVE:
                _df = pd.DataFrame(new_rows)
                combined = pd.concat([existing, _df], ignore_index=True) if len(existing) else _df
                combined.to_parquet(OUT_PARQUET, index=False)
                existing = combined
                new_rows = []

    if new_rows:
        _df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, _df], ignore_index=True) if len(existing) else _df
        combined.to_parquet(OUT_PARQUET, index=False)

    final = pd.read_parquet(OUT_PARQUET)
    elapsed = time.time() - t_start
    print(f"\n=== DONE ===")
    print(f"  total rows: {len(final)}  ok: {final['ok'].sum()}  err: {(~final['ok']).sum()}")
    print(f"  features per recording: {len([c for c in final.columns if c not in ('id','ok','error','duration_sec','wall_sec')])}")
    print(f"  wall: {elapsed/60:.1f} min")
    print(f"  saved: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
