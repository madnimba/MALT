"""
diagnose_gpu_procs.py
---------------------
Run this on the machine to identify what is repeatedly spawning GPU processes.

    python diagnose_gpu_procs.py

Prints: PID, parent PID, user, command line, and uptime for every compute
process currently on the GPU, then prints the full parent process chain
(pstree) for each one so you can see what launched them.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], check=False) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if check and result.returncode != 0:
        print(f"  [stderr]: {result.stderr.strip()}")
    return result.stdout.strip()


def get_gpu_compute_pids(device: int = 0) -> list[int]:
    out = run([
        "nvidia-smi", f"--id={device}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ])
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def proc_info(pid: int) -> dict:
    info = {"pid": pid, "ppid": "?", "user": "?", "cmd": "?", "uptime": "?"}

    # /proc/<pid>/status for ppid and user
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            if line.startswith("PPid:"):
                info["ppid"] = line.split()[1]
            if line.startswith("Uid:"):
                uid = line.split()[1]
                try:
                    import pwd
                    info["user"] = pwd.getpwuid(int(uid)).pw_name
                except Exception:
                    info["user"] = uid

    # /proc/<pid>/cmdline for full command
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if cmdline_path.exists():
        try:
            raw = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            info["cmd"] = raw.strip()[:200]
        except Exception:
            pass

    # /proc/<pid>/stat for start time → uptime
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        try:
            fields = stat_path.read_text().split()
            # field 22 (0-indexed 21) is starttime in clock ticks
            starttime_ticks = int(fields[21])
            hz = os.sysconf("SC_CLK_TCK")
            uptime_secs = float(Path("/proc/uptime").read_text().split()[0])
            proc_age_secs = uptime_secs - (starttime_ticks / hz)
            m, s = divmod(int(proc_age_secs), 60)
            h, m = divmod(m, 60)
            info["uptime"] = f"{h}h {m}m {s}s" if h else f"{m}m {s}s"
        except Exception:
            pass

    return info


def parent_chain(pid: int, depth: int = 0) -> None:
    if depth > 10 or pid <= 1:
        return
    info = proc_info(pid)
    indent = "  " * depth
    print(f"{indent}PID {pid:>7}  [{info['user']}]  {info['cmd'][:120]}")
    try:
        ppid = int(info["ppid"])
    except ValueError:
        return
    if ppid > 1:
        parent_chain(ppid, depth + 1)


def main() -> None:
    own_pid = os.getpid()
    pids = [p for p in get_gpu_compute_pids() if p != own_pid]

    if not pids:
        print("No other compute processes on the GPU right now.")
        return

    print(f"Found {len(pids)} GPU compute process(es): {pids}\n")
    print("=" * 70)

    for pid in pids:
        info = proc_info(pid)
        print(f"\nPID {pid}")
        print(f"  User    : {info['user']}")
        print(f"  Uptime  : {info['uptime']}")
        print(f"  Command : {info['cmd']}")
        print(f"  PPID    : {info['ppid']}")
        print(f"\n  Parent chain (youngest → oldest):")
        try:
            parent_chain(pid, depth=2)
        except Exception as e:
            print(f"    (could not trace: {e})")

        # Also try pstree if available
        pstree_out = run(["pstree", "-s", "-p", str(pid)])
        if pstree_out:
            print(f"\n  pstree -s -p {pid}:")
            for line in pstree_out.splitlines():
                print(f"    {line}")

    print("\n" + "=" * 70)
    print("\nTo stop the spawner, identify the root process in the chain above")
    print("and kill it (or disable the cron job / service that manages it).")
    print("\nQuick kill of just the GPU processes (not the spawner):")
    print(f"  kill -9 {' '.join(str(p) for p in pids)}")


if __name__ == "__main__":
    main()