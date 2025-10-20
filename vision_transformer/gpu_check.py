# save as check_cip_gpu_idle_min.py
import subprocess as sp

# 🔧 改成你的 FAU IDM 账号
USER = "ji18faba"

HOSTS = [
    "cip7b1.cip.cs.fau.de", "cip7b2.cip.cs.fau.de",
    "cip7c0.cip.cs.fau.de", "cip7c1.cip.cs.fau.de", "cip7c2.cip.cs.fau.de",
    "cip7d0.cip.cs.fau.de", "cip7d1.cip.cs.fau.de", "cip7d2.cip.cs.fau.de",
    "cip7e0.cip.cs.fau.de", "cip7e1.cip.cs.fau.de", "cip7e2.cip.cs.fau.de",
    "cip7f0.cip.cs.fau.de", "cip7f1.cip.cs.fau.de", "cip7f2.cip.cs.fau.de",
    "cip7g0.cip.cs.fau.de", "cip7g1.cip.cs.fau.de", "cip7g2.cip.cs.fau.de",
]

CMD = "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits"

# 判定“空闲”的阈值（可按需调整）
IDLE_VRAM_MIB = 600   # 已用显存 <= 600 MiB 视为空闲
IDLE_UTIL_PCT = 10    # GPU 利用率 <= 10% 视为空闲

def ssh_first_line(host: str, timeout: int = 5):
    """返回 (ok, 第一行或错误消息)；只在免密登录已配置的前提下工作。"""
    try:
        p = sp.run(
            [
                "ssh",
                "-o", "BatchMode=yes",          # 禁止密码/交互，没钥匙就直接失败
                "-o", "ConnectTimeout=3",
                "-o", "StrictHostKeyChecking=no",
                "-T",                           # 不分配 TTY
                f"{USER}@{host}",
                CMD,
            ],
            capture_output=True, text=True, timeout=timeout
        )
        if p.returncode == 0 and p.stdout:
            return True, p.stdout.splitlines()[0].strip()
        return False, (p.stderr.strip() or p.stdout.strip() or f"ssh rc={p.returncode}")
    except Exception as e:
        return False, str(e)

def parse(line: str):
    # 形如: "NVIDIA GeForce RTX 3070, 8192, 512, 2"
    parts = [x.strip() for x in line.split(",")]
    name, total, used, util = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
    return name, total, used, util

def check():
    found = 0
    print(f"{'HOST':<22} {'GPU':<22} {'used/total(MiB)':<16} {'util(%)':<7}")
    print("-" * 72)
    for h in HOSTS:
        ok, line = ssh_first_line(h)
        if not ok:
            continue  # 极简：跳过连不上的机器
        try:
            name, total, used, util = parse(line)
        except Exception:
            continue
        if used <= IDLE_VRAM_MIB and util <= IDLE_UTIL_PCT:
            print(f"{h:<22} {name:<22} {used:>4}/{total:<11} {util:>7}")
            found += 1
    if found == 0:
        print("No idle GPUs match the threshold.")

check()
