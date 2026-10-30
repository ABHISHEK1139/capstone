def patch_campaign(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.replace('\r\n', '\n')

    old_rss = '''def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0'''

    new_rss = '''def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    return 0.0'''

    old_timed = '''def timed_trial(fn):
    """Run fn() with SIGALRM cap + RSS sampling. Returns record dict."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TRIAL_TIMEOUT_S)
    baseline = rss_mb()
    sampler = RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        res = fn()
        secs = time.perf_counter() - t0
        sampler.stop()
        peak = max(sampler.max_rss - baseline, 0.0)
        rec = {"status": "ok", "seconds": round(secs, 4),
               "peak_mem_mb": round(peak, 2)}
        rec.update(res)
        return rec
    except RunTimeout as e:
        secs = time.perf_counter() - t0
        sampler.stop()
        return {"status": "timeout", "seconds": round(secs, 4), "error": str(e)}
    except Exception as e:  # noqa: BLE001 - campaign must record, not crash
        secs = time.perf_counter() - t0
        sampler.stop()
        return {"status": "error", "seconds": round(secs, 4),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)'''

    new_timed = '''HAS_SIGALRM = hasattr(signal, "SIGALRM")

def timed_trial(fn):
    """Run fn() with SIGALRM cap (where available) + RSS sampling. Returns record dict."""
    old = None
    if HAS_SIGALRM:
        old = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(TRIAL_TIMEOUT_S)
    baseline = rss_mb()
    sampler = RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    try:
        res = fn()
        secs = time.perf_counter() - t0
        sampler.stop()
        peak = max(sampler.max_rss - baseline, 0.0)
        rec = {"status": "ok", "seconds": round(secs, 4),
               "peak_mem_mb": round(peak, 2)}
        rec.update(res)
        return rec
    except RunTimeout as e:
        secs = time.perf_counter() - t0
        sampler.stop()
        return {"status": "timeout", "seconds": round(secs, 4), "error": str(e)}
    except Exception as e:  # noqa: BLE001 - campaign must record, not crash
        secs = time.perf_counter() - t0
        sampler.stop()
        return {"status": "error", "seconds": round(secs, 4),
                "error": f"{type(e).__name__}: {e}"}
    finally:
        if HAS_SIGALRM:
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)'''

    assert old_rss in content, f'old_rss not found in {filepath}'
    assert old_timed in content, f'old_timed not found in {filepath}'

    content = content.replace(old_rss, new_rss).replace(old_timed, new_timed)
    with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    print(f'Successfully patched {filepath}')

if __name__ == '__main__':
    patch_campaign('benchmarks/run_campaign.py')
