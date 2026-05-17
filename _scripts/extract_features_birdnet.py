"""Phase 1, step 3: BirdNET embedding extraction (1024-dim per 3-sec segment).

Uses encode_arrays with a generator that pre-loads each file at 60-sec cap
via librosa, matching SoundPlot's input data for cross-feature consistency.

Input:  recordings/xeno_canto/XC*.{mp3,wav,flac,ogg}  (1,344 files)
Output: acoustic_features/birdnet_embeddings_v1.parquet
"""
import sys, pathlib, time, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = pathlib.Path(r"D:/Bird Song")
AUDIO_DIR = ROOT / "recordings" / "xeno_canto"
OUT_PARQUET = ROOT / "acoustic_features" / "birdnet_embeddings_v1.parquet"
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)


def main():
    import numpy as np
    import pandas as pd
    import birdnet
    import librosa

    audio_files = sorted([p for p in AUDIO_DIR.glob("XC*.*")
                          if p.suffix.lower() in (".mp3",".wav",".flac",".ogg",".m4a")])
    print(f"audio files: {len(audio_files)}")

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
        return

    print(f"\nloading BirdNET v2.4 ...")
    t0 = time.time()
    model = birdnet.load("acoustic", "2.4", "tf")
    print(f"  loaded in {time.time()-t0:.1f}s  emb_dim={model.get_embeddings_dim()}")

    # Pre-load all 1344 audio files at 60-sec cap (memory cost: ~7 GB at full load)
    # We use a generator to stream, but we also need a parallel ID list since
    # encode_arrays doesn't preserve filename metadata.
    print(f"\npre-loading audio (librosa, 60-sec cap) + encoding ...")
    t0 = time.time()
    arrays = []
    id_list = []
    for i, p in enumerate(to_do):
        try:
            audio, sr = librosa.load(str(p), sr=48000, duration=60.0, mono=True)
            if len(audio) < sr * 0.5:
                continue
            arrays.append((audio, sr))
            id_list.append(p.stem.replace("XC",""))
        except Exception as e:
            print(f"  load failed for {p.name}: {type(e).__name__}: {e}")
        if (i+1) % 200 == 0:
            print(f"  loaded {i+1}/{len(to_do)} ({time.time()-t0:.0f}s)")
    print(f"  loaded {len(arrays)} arrays in {time.time()-t0:.0f}s")

    print(f"\nencoding with BirdNET (n_workers=4, batch=16) ...")
    t_enc = time.time()
    # NOTE: show_stats="minimal" breaks encode_arrays in birdnet 1.x (it tries to
    # treat the audio arrays as file paths in its statistics print path). Omit.
    result = model.encode_arrays(arrays, n_workers=4, batch_size=16)
    elapsed = time.time() - t_enc
    print(f"  encoded in {elapsed/60:.1f}m")

    masked = result.embeddings_masked  # (n_files, max_seg, 1024)
    print(f"  embeddings shape: {masked.shape}")

    new_rows = []
    for i, xc_id in enumerate(id_list):
        e = masked[i]
        valid = ~np.isnan(e).all(axis=1)
        n_seg = int(valid.sum())
        if n_seg == 0:
            new_rows.append({"id": xc_id, "ok": False, "error": "no_segments", "n_segments": 0})
            continue
        v = e[valid]
        mean_vec = np.nanmean(v, axis=0)
        std_vec  = np.nanstd(v, axis=0)
        rec = {"id": xc_id, "ok": True, "error": "", "n_segments": n_seg}
        for k, val in enumerate(mean_vec): rec[f"emb_mean_{k:04d}"] = float(val)
        for k, val in enumerate(std_vec):  rec[f"emb_std_{k:04d}"]  = float(val)
        new_rows.append(rec)

    _df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, _df], ignore_index=True) if len(existing) else _df
    combined.to_parquet(OUT_PARQUET, index=False)
    print(f"\n=== DONE ===  total {len(combined)}  ok {combined['ok'].sum()}  err {(~combined['ok']).sum()}  wall {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
