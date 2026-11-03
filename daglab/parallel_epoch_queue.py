"""Run two independent trials, reverting to serial checkpoint recovery on trouble."""
import re
import subprocess
import time


def memory_free_percent():
    """macOS pressure headroom; unavailable probes do not interrupt experiments."""
    try:
        result = subprocess.run(['/usr/bin/memory_pressure', '-Q'], capture_output=True,
                                text=True, timeout=5)
        match = re.search(r'System-wide memory free percentage:\s*(\d+)%', result.stdout)
        return int(match[1]) if match else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def run_pair(arms, *, launch, complete, stop, wait, event, heartbeat,
             pressure=memory_free_percent, sleep=time.sleep, clock=time.monotonic,
             poll_seconds=5, low_memory_seconds=60, stall_seconds=1800,
             progress_token=lambda arm: None):
    """A failed parallel attempt gets one serial retry; all events are persisted by caller.

    stop must terminate the whole process group, including rollout/plot children.
    Existing successful trials are never rerun. A persistent failure propagates from wait.
    """
    active = {arm: launch(arm) for arm in arms if not complete(arm)}
    low_since = None
    previous = {a: (progress_token(a), clock()) for a in active}
    while active:
        reason = None
        retry = []
        for arm, process in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            if code != 0 or not complete(arm):
                reason = f'{arm} exited with code {code} or without complete summary'
                stop(process)  # also clean up children of an already-dead leader
                retry.append(arm)
            del active[arm]
        if reason is None and len(active) > 1:
            stop_arm = list(active)[-1]
            free = pressure()
            now = clock()
            low_since = (now if low_since is None else low_since) if free is not None and free < 10 else None
            if low_since is not None and now - low_since >= low_memory_seconds:
                reason = f'memory headroom below 10% for {low_memory_seconds}s (now {free}%)'
            for arm in active:
                token = progress_token(arm)
                old, changed = previous[arm]
                previous[arm] = (token, now) if token != old else (old, changed)
                if now - previous[arm][1] >= stall_seconds:
                    reason = f'{arm} progress unchanged for {stall_seconds}s'
                    stop_arm = arm
            if reason:
                arm = stop_arm
                stop(active.pop(arm))
                retry.append(arm)
        if reason:
            event(dict(status='serial_fallback', reason=reason, retry=retry,
                       running=list(active)))
            # Wait for healthy work first; a second failure also receives one serial retry.
            for arm, process in active.items():
                try:
                    wait(process)
                    assert complete(arm), f'{arm}: missing complete summary'
                except Exception as error:
                    stop(process)
                    retry.append(arm)
                    event(dict(status='serial_retry_queued', arm=arm, error=str(error)))
            for arm in retry:
                event(dict(status='serial_retry_started', arm=arm))
                wait(launch(arm))
                assert complete(arm), f'{arm}: serial retry incomplete'
            return 'serial_fallback'
        heartbeat()
        if active:
            sleep(poll_seconds)
    return 'parallel_complete'
