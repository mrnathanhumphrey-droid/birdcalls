"""Phrase-level rhythm templates — Angle 2.

For each of the 60 song-sparrow + marsh-wren recordings already extracted,
segment the song into PHRASES (groups of onsets separated by gaps > GAP_S)
and encode each phrase as a feature triple.

A phrase = a coherent burst of syllables with short internal gaps.
Inter-phrase gaps are usually 2-10× longer than within-phrase IOIs.

Phrase features:
  duration_sec   — total time of the phrase
  n_syllables    — count of onsets within the phrase
  mean_iio       — mean inter-onset interval inside the phrase
  mean_intensity — mean RMS or onset envelope inside the phrase
  intensity_max  — peak onset envelope

A recording = ordered sequence of phrase-feature-tuples = a "phrase template."

Two recordings' templates compare via DTW on the phrase sequences.

This is more aggregated than per-frame trajectory DTW. Birdsong literature
finds phrase-template structure to be the most stereotyped level of song.
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
OUT_JSON = ROOT / "acoustic_features" / "phrase_templates_v1.json"

TARGET_SR = 22050
HOP_MS = 25
HOP_LENGTH = int(TARGET_SR * HOP_MS / 1000)
N_FFT = 1024

# A "phrase break" = inter-onset gap longer than this many SECONDS
PHRASE_GAP_S = 0.5

# Species + BCR cells we already extracted trajectories for
SPECIES_LIST = ["song sparrow", "marsh wren"]
TARGET_BCRS = ["COASTAL_CALIFORNIA", "SONORAN_AND_MOJAVE_DESERTS",
               "NEW_ENGLAND/MID-ATLANTIC_COAST"]
N_PER_BCR = 10


def to_sec(s):
    try:
        if isinstance(s, str) and ":" in s:
            m, ss = s.split(":")
            return int(m) * 60 + int(ss)
        return float(s)
    except: return float("nan")


def pick_recordings():
    meta = pd.read_parquet(META)
    out = []
    for sp in SPECIES_LIST:
        sub = meta[(meta["common_name"] == sp) & (meta["BCR_NAME"].isin(TARGET_BCRS))].copy()
        sub = sub[sub["q_score"].isin(["A", "B"])]
        sub["length_sec"] = sub["length"].apply(to_sec)
        sub = sub[(sub["length_sec"] >= 3) & (sub["length_sec"] <= 60)]
        for bcr in TARGET_BCRS:
            bcr_sub = sub[sub["BCR_NAME"] == bcr].sort_values("q_score").head(N_PER_BCR * 4)
            picked = []
            for _, r in bcr_sub.iterrows():
                cands = list(AUDIO_DIR.glob(f"XC{r['id']}.*"))
                if cands:
                    r2 = r.to_dict()
                    r2["audio_path"] = str(cands[0])
                    r2["species"] = sp
                    picked.append(r2)
                if len(picked) >= N_PER_BCR: break
            out.extend(picked)
    return out


def extract_phrases(audio_path, sp, bcr, xc_id):
    """Detect onsets, group into phrases, encode each phrase as feature tuple."""
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, duration=30.0)
    if len(y) < TARGET_SR * 0.5: return None
    y = y / (np.max(np.abs(y)) + 1e-9)

    onset_env = librosa.onset.onset_strength(y=y, sr=TARGET_SR, hop_length=HOP_LENGTH,
                                              n_fft=N_FFT, aggregate=np.median)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=TARGET_SR,
                                         hop_length=HOP_LENGTH, units="time", backtrack=False)
    if len(onsets) < 2:
        return None

    # Group onsets into phrases by gap threshold
    phrases = []  # list of (onset_times_array_within_phrase)
    cur = [onsets[0]]
    for t in onsets[1:]:
        if t - cur[-1] > PHRASE_GAP_S:
            phrases.append(np.array(cur)); cur = [t]
        else:
            cur.append(t)
    if cur: phrases.append(np.array(cur))

    # Per-phrase features
    phrase_feats = []
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    rms_times = np.arange(len(rms)) * HOP_LENGTH / TARGET_SR
    for ph in phrases:
        if len(ph) == 0: continue
        ph_start, ph_end = float(ph[0]), float(ph[-1])
        duration = ph_end - ph_start + 0.05  # tiny pad
        n_syll = len(ph)
        if n_syll >= 2:
            mean_iio = float(np.mean(np.diff(ph)))
        else:
            mean_iio = 0.0
        # Intensity within phrase span
        ph_mask = (rms_times >= ph_start - 0.05) & (rms_times <= ph_end + 0.05)
        if ph_mask.any():
            mean_int = float(rms[ph_mask].mean())
            max_int = float(rms[ph_mask].max())
        else:
            mean_int = max_int = 0.0
        phrase_feats.append({
            "start": ph_start, "end": ph_end,
            "duration": duration, "n_syllables": n_syll,
            "mean_iio": mean_iio,
            "mean_intensity": mean_int,
            "max_intensity": max_int,
        })

    return {
        "id": xc_id,
        "species": sp,
        "bcr": bcr,
        "duration_sec": float(len(y) / TARGET_SR),
        "n_total_onsets": int(len(onsets)),
        "n_phrases": len(phrase_feats),
        "phrases": phrase_feats,
    }


def main():
    recs = pick_recordings()
    print(f"=== {len(recs)} recordings selected ===")

    templates = []
    for r in recs:
        t = extract_phrases(r["audio_path"], r["species"], r["BCR_NAME"], str(r["id"]))
        if t is None: continue
        templates.append(t)
        print(f"  XC{r['id']:<8s} {r['species']:<15s} {r['BCR_NAME']:<35s} "
              f"phrases={t['n_phrases']:2d} onsets={t['n_total_onsets']:3d}")

    payload = {
        "schema_version": "phrase_templates_v1",
        "species_list": SPECIES_LIST,
        "bcrs": TARGET_BCRS,
        "n_per_bcr": N_PER_BCR,
        "phrase_gap_threshold_sec": PHRASE_GAP_S,
        "feature_keys": ["duration", "n_syllables", "mean_iio", "mean_intensity", "max_intensity"],
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_templates": len(templates),
        "templates": templates,
    }
    OUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"\n  -> {OUT_JSON.relative_to(ROOT)}  ({OUT_JSON.stat().st_size:,} bytes)")

    # Summary
    print(f"\n  === phrase distribution per (species, BCR) ===")
    rows = []
    for t in templates:
        rows.append({"species": t["species"], "bcr": t["bcr"], "n_phrases": t["n_phrases"]})
    summary = pd.DataFrame(rows).groupby(["species", "bcr"]).agg(
        n_recs=("n_phrases", "count"),
        mean_phrases=("n_phrases", "mean"),
        std_phrases=("n_phrases", "std"),
    ).reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
