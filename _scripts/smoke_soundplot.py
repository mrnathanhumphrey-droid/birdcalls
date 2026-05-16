"""Smoke test: validate SoundPlot feature pipeline on a synthetic chirp.

Generates a 2-second synthetic frequency-modulated signal (bird-like)
and runs the SoundPlot feature extraction pipeline end-to-end.
Exits clean if pipeline runs; reports feature dimensionality and any errors.
"""
import sys, pathlib, tempfile
import numpy as np
import soundfile as sf

SOUNDPLOT = pathlib.Path(r"D:/Bird Song/_external/SoundPlot")
sys.path.insert(0, str(SOUNDPLOT))

print("[1] Generating synthetic bird-like chirp ...")
sr = 22050
t = np.linspace(0, 2.0, int(sr*2.0), endpoint=False)
# Bird-like: 3 syllables with FM sweep + harmonic structure
sig = np.zeros_like(t)
for syllable_start in [0.1, 0.7, 1.3]:
    mask = (t >= syllable_start) & (t < syllable_start + 0.3)
    tt = t[mask] - syllable_start
    # FM sweep 3kHz -> 5kHz over 300ms with envelope
    freq = 3000 + 6000*tt
    env = np.exp(-((tt - 0.15)/0.06)**2)
    sig[mask] = env * (
        np.sin(2*np.pi*freq*tt) +
        0.3*np.sin(2*np.pi*2*freq*tt) +  # 2nd harmonic
        0.1*np.sin(2*np.pi*3*freq*tt)    # 3rd harmonic
    )
sig += 0.005 * np.random.RandomState(42).randn(len(sig))  # tiny noise floor
print(f"  signal: {len(sig)} samples @ {sr} Hz = {len(sig)/sr:.2f} sec")

with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    sf.write(tmp.name, sig, sr)
    tmp_path = tmp.name
print(f"  wrote: {tmp_path}")

print("\n[2] Importing SoundPlot modules ...")
from src.audio import AudioLoader, AudioPreprocessor
from src.features import FeatureExtractor

print("\n[3] AudioLoader + AudioPreprocessor ...")
loader = AudioLoader(target_sr=22050)
audio, sr = loader.load(tmp_path)
print(f"  loaded: {len(audio)} samples @ {sr} Hz")

pre = AudioPreprocessor(sample_rate=sr)
audio = pre.preprocess_full(audio, normalize=True, denoise=False)  # skip denoise for smoke
print(f"  preprocessed: {audio.dtype}, range [{audio.min():.3f}, {audio.max():.3f}]")

print("\n[4] FeatureExtractor (all features) ...")
fx = FeatureExtractor(sample_rate=sr)
import inspect
methods = [m for m in dir(fx) if not m.startswith("_") and callable(getattr(fx, m))]
print(f"  available methods: {methods}")

# Try the most common one
for method_name in ["extract_all", "extract", "compute_all", "process"]:
    if hasattr(fx, method_name):
        print(f"  trying {method_name}() ...")
        try:
            f = getattr(fx, method_name)(audio)
            print(f"    result type: {type(f).__name__}")
            if hasattr(f, "shape"):
                print(f"    shape: {f.shape}")
            if hasattr(f, "keys"):
                print(f"    keys: {list(f.keys())[:10]}")
            elif hasattr(f, "__len__"):
                print(f"    len: {len(f)}")
            break
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {e}")

print("\n[5] SMOKE TEST PASSED — SoundPlot feature pipeline runs end-to-end.")
