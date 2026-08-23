# -*- coding: utf-8 -*-
"""
download_data_a6000.py — RTX A6000 新机器一键部署脚本（Windows 可用）

从网上下载 StormSurgeCastNet-SDA 训练/评估所需的全部内容，并用 SHA256SUMS.txt 校验：

  1. 代码   : GitHub 仓库（已在线）
  2. 模型   : GitHub Release 资产（best_sda.pth.tar / last.pth.tar，上传后在线）
  3. 缓存   : Zenodo 记录（cache_sda_real256 / cache_sda_full / cache_gulf，上传后在线）

用法（在新机器 A6000_code/ 目录内运行）:
  python download_data_a6000.py --code-only                          # 只下代码（GitHub zip）
  python download_data_a6000.py --models <url1> <url2> ...           # 下模型（Release 资产直链）
  python download_data_a6000.py --caches --zenodo <记录ID>           # 下缓存（Zenodo 记录）
  python download_data_a6000.py --caches --zenodo <记录ID> --only cache_sda_real256
  python download_data_a6000.py --verify                            # 仅校验已下载文件
"""
import os, sys, io, json, hashlib, urllib.request, zipfile, argparse, time

REPO = "LEEchina-star/StormSurgeCastNet-SDA"
REPO_ZIP = f"https://github.com/{REPO}/archive/refs/heads/main.zip"
SUMS = "SHA256SUMS.txt"   # 校验清单（含相对路径）

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def load_sums():
    """SHA256SUMS.txt: '<hash>  <size>  <relpath>' -> dict relpath -> (hash, size)"""
    d = {}
    if os.path.isfile(SUMS):
        for line in open(SUMS, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            h, size, rel = line.split(None, 2)
            d[rel] = (h, int(size))
    return d

def download(url, dest, expected_size=None):
    os.makedirs(os.path.dirname(dest), exist_ok=True) if os.path.dirname(dest) else None
    if os.path.isfile(dest) and expected_size and os.path.getsize(dest) == expected_size:
        print(f"  已存在，跳过: {dest}")
        return True
    print(f"  下载 {url} -> {dest}")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done / total * 100
                sys.stdout.write(f"\r    {pct:5.1f}%  ({done/1e6:.0f}/{total/1e6:.0f} MB)")
                sys.stdout.flush()
    print(f"\n    完成 ({time.time()-t0:.0f}s)")
    return True

def verify(rel):
    if not os.path.isfile(rel):
        print(f"  [缺失] {rel}")
        return False
    h, size = SUMS_DICT[rel]
    if os.path.getsize(rel) != size:
        print(f"  [大小不符] {rel} ({os.path.getsize(rel)} != {size})")
        return False
    print(f"  校验 {rel} ...", end="", flush=True)
    ok = sha256_file(rel) == h
    print(" OK" if ok else " [哈希不符!]")
    return ok

def cmd_code(args):
    print(f"== 1. 下载代码 {REPO_ZIP}")
    download(REPO_ZIP, "repo.zip")
    print("解压 repo.zip -> 覆盖当前目录...")
    with zipfile.ZipFile("repo.zip") as z:
        names = z.namelist()
        rootdir = names[0].split("/")[0]
        for n in names:
            parts = n.split("/", 1)
            if len(parts) == 2 and parts[1]:
                z.extract(n)
    print(f"解压完成（顶层目录 {rootdir}/，把 A6000_code/* 拷到当前目录或直接在其内使用）")

def cmd_models(args):
    print(f"== 2. 下载模型权重（{len(args.urls)} 个）")
    os.makedirs("models_upload", exist_ok=True)
    for u in args.urls:
        fn = u.split("/")[-1].split("?")[0]
        download(u, os.path.join("models_upload", fn))
    print("模型已放入 models_upload/（训练时用 --checkpoint 指定，或拷到 results_sda/<name>/ 下）")

def cmd_caches(args):
    if not args.zenodo:
        print("请提供 Zenodo 记录 ID：--zenodo <ID>（上传缓存后获得）")
        sys.exit(1)
    print(f"== 3. 从 Zenodo 记录 {args.zenodo} 下载缓存")
    api = f"https://zenodo.org/api/records/{args.zenodo}"
    with urllib.request.urlopen(api, timeout=60) as r:
        rec = json.load(r)
    files = rec.get("files", [])
    want = args.only
    for f in files:
        name = f["key"]
        if want and want not in name:
            continue
        link = f.get("links", {}).get("self") or f.get("links", {}).get("link")
        if not link:
            link = f"https://zenodo.org/records/{args.zenodo}/files/{urllib.parse.quote(name)}?download=1"
        download(link, name, f.get("size"))

def cmd_verify(args):
    print("== 校验已下载文件（对照 SHA256SUMS.txt）")
    bad = 0
    for rel in SUMS_DICT:
        if os.path.isfile(rel):
            if not verify(rel):
                bad += 1
    print(f"校验完成：{'全部 OK' if bad == 0 else f'{bad} 个文件异常'}")

SUMS_DICT = {}
ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd")
p1 = sub.add_parser("code-only"); p1.set_defaults(fn=cmd_code)
p2 = sub.add_parser("models"); p2.add_argument("urls", nargs="+"); p2.set_defaults(fn=cmd_models)
p3 = sub.add_parser("caches"); p3.add_argument("--zenodo"); p3.add_argument("--only"); p3.set_defaults(fn=cmd_caches)
p4 = sub.add_parser("verify"); p4.set_defaults(fn=cmd_verify)
args = ap.parse_args()
if not args.cmd:
    ap.print_help(); sys.exit(1)
SUMS_DICT = load_sums()
print(f"SHA256SUMS.txt 已加载 {len(SUMS_DICT)} 个文件条目")
args.fn(args)
