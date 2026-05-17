"""Proper structural features per pre-reg spec.

For each recording:
1. Onset-segment into syllables (librosa onsets)
2. For each syllable, compute spectral signature (MFCC mean)
3. Cluster all syllables across ALL recordings of a species into syllable-types (HDBSCAN)
4. Per recording, compute:
   - syllable_inventory_size: distinct syllable-types used
   - transition_entropy: Shannon entropy of bigram transitions between syllable-types
   - lz_complexity: Lempel-Ziv (LZ77-style) compression ratio of the syllable sequence
   - n_syllables: total syllables

Output: acoustic_features/structural_features_v1.parquet
"""
import os, sys, pathlib, time, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
import numpy as np

ROOT = pathlib.Path(r"D:/Bird Song")
AUDIO_DIR = ROOT / "recordings" / "xeno_canto"
OUT_PARQUET = ROOT / "acoustic_features" / "structural_features_v1.parquet"
META = ROOT / "range_data" / "xc_metadata_joined_bcr.parquet"


def segment_and_mfcc(audio_path_str):
    """Worker: onset-segment audio into syllables, compute MFCC mean per syllable.
    Returns (xc_id, common_name placeholder, [(start_s, end_s, mfcc_vec[13]), ...])
    """
    import warnings; warnings.filterwarnings("ignore")
    import librosa, numpy as np, pathlib
    p = pathlib.Path(audio_path_str)
    xc_id = p.stem.replace("XC", "")
    try:
        y, sr = librosa.load(str(p), sr=22050, duration=60.0, mono=True)
        if len(y) < sr * 0.5:
            return (xc_id, [])
        # Onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples",
                                            backtrack=True, hop_length=512)
        if len(onsets) < 2:
            return (xc_id, [])
        # Build syllable boundaries: onset_i to onset_{i+1} (or end of audio)
        bounds = list(zip(onsets[:-1], onsets[1:]))
        bounds.append((onsets[-1], len(y)))
        syllables = []
        for s, e in bounds:
            if e - s < int(0.03 * sr):  # skip syllables shorter than 30ms
                continue
            if e - s > int(2.0 * sr):  # skip "syllables" longer than 2s (likely silence-bridged)
                continue
            syl = y[s:e]
            mfcc = librosa.feature.mfcc(y=syl, sr=sr, n_mfcc=13)
            syllables.append((s/sr, e/sr, mfcc.mean(axis=1).tolist()))
        return (xc_id, syllables)
    except Exception as ex:
        return (xc_id, [])


def lz_complexity(seq):
    """Lempel-Ziv complexity (number of distinct substrings) on a list of ints."""
    if not seq: return 0
    s = [str(x) for x in seq]
    # Count distinct prefixes in the LZ78 parsing
    dict_ = set()
    i, n = 0, len(s)
    count = 0
    while i < n:
        # Find longest prefix already in dict
        j = i
        while j < n:
            sub = "_".join(s[i:j+1])
            if sub not in dict_:
                dict_.add(sub)
                count += 1
                i = j + 1
                break
            j += 1
        else:
            i = n
    return count


def main():
    import pandas as pd
    import hdbscan
    meta = pd.read_parquet(META)
    meta["id"] = meta["id"].astype(str)
    audio_files = sorted([p for p in AUDIO_DIR.glob("XC*.*")
                          if p.suffix.lower() in (".mp3",".wav",".flac",".ogg",".m4a")])
    print(f"audio files: {len(audio_files)}")
    id_to_species = dict(zip(meta["id"].astype(str), meta["common_name"]))

    # Phase A: parallel syllable segmentation + per-syllable MFCC
    print(f"\n[A] Onset-segmenting + per-syllable MFCC (12 workers, ~10-30 min) ...")
    t0 = time.time()
    n_workers = min(12, mp.cpu_count() - 2)
    syllables_by_id = {}  # xc_id -> [(start, end, mfcc[13]), ...]
    done = 0
    with mp.Pool(n_workers) as pool:
        for xc_id, syls in pool.imap_unordered(segment_and_mfcc, [str(p) for p in audio_files], chunksize=8):
            syllables_by_id[xc_id] = syls
            done += 1
            if done % 100 == 0:
                print(f"  segmented {done}/{len(audio_files)}  ({(time.time()-t0)/60:.1f}m)")
    print(f"  done segmenting in {(time.time()-t0)/60:.1f} min")
    print(f"  total syllables: {sum(len(s) for s in syllables_by_id.values())}")

    # Phase B: per-species, cluster syllables into types (UMAP->8d + HDBSCAN per pre-reg spec)
    print(f"\n[B] Per-species syllable-type clustering (UMAP->8d + HDBSCAN) ...")
    import umap
    species_syl_types = {}  # (xc_id, syl_idx) -> type_label
    for species in sorted({id_to_species.get(xid, "unknown") for xid in syllables_by_id.keys() if xid in id_to_species}):
        if species == "unknown": continue
        all_mfccs = []
        all_keys = []
        for xc_id, syls in syllables_by_id.items():
            if id_to_species.get(xc_id) != species: continue
            for i, (s_, e_, mfcc) in enumerate(syls):
                all_mfccs.append(mfcc)
                all_keys.append((xc_id, i))
        if len(all_mfccs) < 50:
            print(f"  {species}: only {len(all_mfccs)} syllables, skipping")
            continue
        X = np.array(all_mfccs)
        # UMAP to 8 dims (per pre-reg)
        rs = 20260516
        reducer = umap.UMAP(n_components=8, n_neighbors=15, min_dist=0.0,
                            metric="euclidean", random_state=rs, low_memory=True)
        Xr = reducer.fit_transform(X)
        # HDBSCAN with hard-floor min_cluster_size = 15 (not % of total)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=3,
                                    cluster_selection_method="eom")
        labels = clusterer.fit_predict(Xr)
        n_types = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        print(f"  {species}: {len(X)} syllables -> {n_types} types + {n_noise} noise ({100*n_noise/len(X):.1f}%)")
        for k, lab in zip(all_keys, labels):
            species_syl_types[k] = int(lab)

    # Phase C: per-recording structural features
    print(f"\n[C] Per-recording structural feature aggregation ...")
    rows = []
    for xc_id, syls in syllables_by_id.items():
        types = [species_syl_types.get((xc_id, i), -1) for i in range(len(syls))]
        types_clean = [t for t in types if t >= 0]  # drop noise syllables
        n_syl = len(types_clean)
        if n_syl == 0:
            rows.append({"id": xc_id, "n_syllables": 0,
                         "syllable_inventory_size": 0,
                         "transition_entropy": 0.0,
                         "lz_complexity": 0,
                         "lz_ratio": 0.0})
            continue
        inv = len(set(types_clean))
        # Transition entropy (bigrams)
        if len(types_clean) >= 2:
            from collections import Counter
            bigrams = Counter(zip(types_clean[:-1], types_clean[1:]))
            total = sum(bigrams.values())
            probs = np.array(list(bigrams.values())) / total
            entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        else:
            entropy = 0.0
        lzc = lz_complexity(types_clean)
        lz_ratio = lzc / n_syl if n_syl > 0 else 0.0
        rows.append({
            "id": xc_id, "n_syllables": n_syl,
            "syllable_inventory_size": inv,
            "transition_entropy": entropy,
            "lz_complexity": lzc,
            "lz_ratio": lz_ratio,
        })
    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\n=== DONE ===")
    print(df.describe().to_string())
    print(f"\nsaved: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
