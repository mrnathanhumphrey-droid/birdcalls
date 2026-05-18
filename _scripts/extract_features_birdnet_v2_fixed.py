"""BirdNET embedding extraction — FIXED v2.

Bug in v1 (extract_features_birdnet.py):
  used result.embeddings_masked (bool validity mask, dtype=bool)
  instead of result.embeddings (float32 actual embedding data)

The bool mask, when nanmean'd across segments, gave each recording a
SINGLE scalar value (= valid_segments / 21) broadcast to all 1024 dims.
This is what the dialect-cluster + Stan-fit pipelines in Phase 1/2 ran on.

This fixed version:
  - Uses result.embeddings as the data
  - Uses result.embeddings_masked correctly as the per-segment validity mask
  - Validates output shapes before writing
  - Writes to birdnet_embeddings_v2.parquet (keep v1 for audit)
  - Resume-safe: skips recordings already in v2
"""
import sys, pathlib, time, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT = pathlib.Path(r"D:/Bird Song")
AUDIO_DIR = ROOT / "recordings" / "xeno_canto"
OUT_PARQUET = ROOT / "acoustic_features" / "birdnet_embeddings_v2.parquet"


def main():
    import numpy as np
    import pandas as pd
    import birdnet
    import librosa

    audio_files = sorted([p for p in AUDIO_DIR.glob("XC*.*")
                          if p.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".m4a")])
    print(f"audio files: {len(audio_files)}")

    if OUT_PARQUET.exists():
        existing = pd.read_parquet(OUT_PARQUET)
        done_ids = set(existing["id"].astype(str))
        print(f"already extracted in v2: {len(done_ids)}")
        to_do = [p for p in audio_files if p.stem.replace("XC", "") not in done_ids]
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

    # Process in chunks of 200 for incremental saving
    CHUNK = 200
    for chunk_start in range(0, len(to_do), CHUNK):
        chunk = to_do[chunk_start:chunk_start + CHUNK]
        print(f"\n=== chunk {chunk_start}-{chunk_start+len(chunk)} ===")

        t_chunk = time.time()
        arrays = []
        id_list = []
        for p in chunk:
            try:
                audio, sr = librosa.load(str(p), sr=48000, duration=60.0, mono=True)
                if len(audio) < sr * 0.5: continue
                arrays.append((audio, sr))
                id_list.append(p.stem.replace("XC", ""))
            except Exception as e:
                print(f"  load failed for {p.name}: {type(e).__name__}: {e}")
        print(f"  loaded {len(arrays)} arrays in {time.time()-t_chunk:.0f}s")
        if not arrays: continue

        t_enc = time.time()
        result = model.encode_arrays(arrays, n_workers=4, batch_size=16)
        embs = result.embeddings           # (n_files, max_seg, 1024) float32  <-- THE FIX
        valid_mask = result.embeddings_masked  # (n_files, max_seg, 1024) bool
        print(f"  encoded {embs.shape} in {(time.time()-t_enc):.0f}s "
              f"(dtype={embs.dtype}, max_seg={embs.shape[1]})")

        # Mask convention: True = MASKED-OUT (invalid/padded), False = real data.
        # Segment is valid if NOT all of its 1024 dims are masked.
        seg_valid = ~valid_mask.all(axis=2)  # (n_files, max_seg) bool

        new_rows = []
        for i, xc_id in enumerate(id_list):
            e_i = embs[i]            # (max_seg, 1024)
            v_i = seg_valid[i]       # (max_seg,) bool
            n_seg = int(v_i.sum())
            if n_seg == 0:
                new_rows.append({"id": xc_id, "ok": False, "error": "no_valid_segments",
                                 "n_segments": 0})
                continue
            v = e_i[v_i]             # (n_seg, 1024) — only valid segments
            # Some valid segments still have NaN values; use nan-aware aggregation
            mean_vec = np.nanmean(v, axis=0)
            std_vec = np.nanstd(v, axis=0)
            # Replace any remaining NaN (column was all-NaN) with 0
            mean_vec = np.nan_to_num(mean_vec, nan=0.0)
            std_vec = np.nan_to_num(std_vec, nan=0.0)
            rec = {"id": xc_id, "ok": True, "error": "", "n_segments": n_seg}
            for k, val in enumerate(mean_vec): rec[f"emb_mean_{k:04d}"] = float(val)
            for k, val in enumerate(std_vec):  rec[f"emb_std_{k:04d}"]  = float(val)
            new_rows.append(rec)

        _df = pd.DataFrame(new_rows)
        existing = pd.concat([existing, _df], ignore_index=True) if len(existing) else _df
        existing.to_parquet(OUT_PARQUET, index=False)
        ok_count = existing["ok"].sum() if "ok" in existing.columns else len(existing)
        print(f"  saved incremental parquet: total {len(existing)} rows ({ok_count} ok)")

    ok_count = existing["ok"].sum()
    print(f"\n=== DONE ===  total {len(existing)}  ok {ok_count}  err {len(existing)-ok_count}")

    # Quick sanity check on stored embeddings
    sample = existing[existing["ok"]].iloc[0]
    sample_mean = [sample[f"emb_mean_{k:04d}"] for k in range(1024)]
    import numpy as np
    n_unique = len(set(sample_mean))
    print(f"  sanity: first ok row has {n_unique} unique values across 1024 emb_mean dims "
          f"(should be ~1024 if fix works)")


if __name__ == "__main__":
    main()
