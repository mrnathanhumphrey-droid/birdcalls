"""Watchdog for BirdNET embedding extraction.

Identifies OUR process tree (matched by cmdline containing 'extract_features_birdnet.py'),
monitors stdout log + parquet for progress, kills + refires if stalled. Never
touches non-matching processes (cancer, server, etc.).

Stall criterion: no growth in stdout log AND no parquet update for STALL_MIN minutes.
On stall: kill our process tree (children too), confirm dead, refire BirdNET.
Max retries: 3.

Run with: python _scripts/watchdog_birdnet.py
"""
import os, sys, time, pathlib, subprocess, datetime
import psutil

ROOT = pathlib.Path(r"D:/Bird Song")
TARGET_SCRIPT = "extract_features_birdnet.py"
OUT_PARQUET = ROOT / "acoustic_features" / "birdnet_embeddings_v1.parquet"
STDOUT_LOG  = ROOT / "watchdog_birdnet_stdout.log"
WATCHDOG_LOG = ROOT / "watchdog_birdnet.log"

STALL_MIN = 30        # minutes without progress -> stalled
CHECK_SEC = 60        # poll interval
MAX_RETRIES = 3
STARTUP_GRACE_SEC = 60  # wait this long after firing before stall checks


def log(msg):
    line = f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}"
    with open(WATCHDOG_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def find_our_process():
    """Find the python process running our target script (NOT the spawn workers).
    Returns psutil.Process or None.
    """
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = p.info["cmdline"] or []
            cmdline_str = " ".join(str(c) for c in cmdline)
            name = (p.info["name"] or "").lower()
            if "python" in name and TARGET_SCRIPT in cmdline_str:
                # Exclude multiprocessing-spawn workers (they have 'multiprocessing.spawn')
                if "multiprocessing.spawn" in cmdline_str:
                    continue
                # Exclude THIS watchdog process and any 'python -c' diagnostics
                if "-c" in cmdline and any("import psutil" in str(c) for c in cmdline):
                    continue
                return psutil.Process(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def kill_our_tree(p):
    """Kill p + all descendants. Returns count killed. SAFE — only kills our subtree."""
    killed = []
    try:
        children = p.children(recursive=True)
        log(f"  killing PID {p.pid} + {len(children)} descendants")
        for c in children:
            try:
                log(f"    child PID {c.pid}: {c.name()}")
                c.kill()
                killed.append(c)
            except psutil.NoSuchProcess:
                pass
        try:
            p.kill()
            killed.append(p)
        except psutil.NoSuchProcess:
            pass
        gone, alive = psutil.wait_procs(killed, timeout=10)
        log(f"  killed: {len(gone)}, still alive: {len(alive)}")
        for survivor in alive:
            try:
                log(f"  FORCE kill survivor PID {survivor.pid}")
                survivor.terminate()
            except: pass
        return len(gone)
    except Exception as e:
        log(f"  kill exception: {e}")
        return len(killed)


def fire_birdnet():
    """Spawn a fresh BirdNET extraction process. Returns subprocess.Popen."""
    log(f"Firing python -X utf8 _scripts/{TARGET_SCRIPT} ...")
    # Open in append mode so we keep history across retries
    stdout = open(STDOUT_LOG, "a", buffering=1)
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", f"_scripts/{TARGET_SCRIPT}"],
        cwd=str(ROOT),
        stdout=stdout,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    log(f"  fired PID {proc.pid}")
    return proc


def safe_size(path):
    try: return path.stat().st_size
    except: return 0

def safe_mtime(path):
    try: return path.stat().st_mtime
    except: return 0


def main():
    log("=" * 60)
    log(f"=== Watchdog START ===")
    log(f"  target: {TARGET_SCRIPT}")
    log(f"  output: {OUT_PARQUET}")
    log(f"  stdout log: {STDOUT_LOG}")
    log(f"  stall threshold: {STALL_MIN} min  check interval: {CHECK_SEC} s")
    log(f"  startup grace: {STARTUP_GRACE_SEC} s  max retries: {MAX_RETRIES}")

    # If parquet already exists and is non-trivial, we're done
    if OUT_PARQUET.exists() and safe_size(OUT_PARQUET) > 10 * 1024 * 1024:
        log(f"  parquet already complete ({safe_size(OUT_PARQUET)//1024//1024} MB). Nothing to do.")
        return

    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        log(f"\n--- Attempt {attempt}/{MAX_RETRIES} ---")

        # Check if BirdNET is already running (e.g., user fired it manually)
        existing = find_our_process()
        if existing is not None:
            log(f"  found existing PID {existing.pid} — watching it instead of firing new")
            proc_pid = existing.pid
        else:
            proc = fire_birdnet()
            proc_pid = proc.pid
            log(f"  startup grace: sleeping {STARTUP_GRACE_SEC}s")
            time.sleep(STARTUP_GRACE_SEC)

        # Initial state
        baseline_log_size = safe_size(STDOUT_LOG)
        baseline_parquet_mtime = safe_mtime(OUT_PARQUET)
        last_progress_ts = time.time()
        last_log_size = baseline_log_size
        last_parquet_mtime = baseline_parquet_mtime
        log(f"  baseline: log_size={baseline_log_size}  parquet_mtime={baseline_parquet_mtime}")

        # Watch loop
        while True:
            time.sleep(CHECK_SEC)

            # Refind process (PID may have rotated if multiprocessing root changed)
            try:
                p = psutil.Process(proc_pid)
                running = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                running = False

            if not running:
                # Process gone. Did it succeed?
                parquet_size = safe_size(OUT_PARQUET)
                if parquet_size > 10 * 1024 * 1024:
                    log(f"  Process exited. Parquet = {parquet_size//1024//1024} MB. SUCCESS.")
                    return
                else:
                    log(f"  Process exited. Parquet = {parquet_size} bytes. CRASH — will retry.")
                    break  # to outer loop for retry

            # Process alive — check progress
            current_log_size = safe_size(STDOUT_LOG)
            current_parquet_mtime = safe_mtime(OUT_PARQUET)
            progress_made = (current_log_size > last_log_size) or (current_parquet_mtime > last_parquet_mtime)

            if progress_made:
                last_progress_ts = time.time()
                # Get tree stats
                try:
                    kids = p.children(recursive=True)
                    total_rss_mb = (p.memory_info().rss + sum(k.memory_info().rss for k in kids)) / 1024 / 1024
                    log(f"  alive (PID {proc_pid} + {len(kids)} kids), log={current_log_size}B, "
                        f"parq_mtime={current_parquet_mtime}, total_rss={total_rss_mb:.0f}MB")
                except psutil.NoSuchProcess:
                    pass
                last_log_size = current_log_size
                last_parquet_mtime = current_parquet_mtime
            else:
                mins_idle = (time.time() - last_progress_ts) / 60
                log(f"  no progress for {mins_idle:.1f} min")
                if mins_idle > STALL_MIN:
                    log(f"  STALL DETECTED (idle > {STALL_MIN} min)")
                    kill_our_tree(p)
                    time.sleep(5)
                    survivor = find_our_process()
                    if survivor is None:
                        log(f"  confirmed dead. Will refire.")
                    else:
                        log(f"  WARN: survivor PID {survivor.pid} found. Killing again.")
                        kill_our_tree(survivor)
                    break  # retry outer loop

    log(f"\n=== Watchdog EXHAUSTED {MAX_RETRIES} retries. Manual intervention needed. ===")


if __name__ == "__main__":
    main()
