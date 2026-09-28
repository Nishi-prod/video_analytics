#!/usr/bin/env python3
"""Download the public benchmark datasets listed in manifest.json.

Run this on the team server (not a laptop): several datasets are 5-70 GB.

    python datasets/download.py list
    python datasets/download.py get mot17 mot20 caviar --root /data/rvb
    python datasets/download.py get smartspaces --root /data/rvb [--with-videos]
    python datasets/download.py get meva --prefix <folder>/ --root /data/rvb
    python datasets/download.py verify --root /data/rvb
    python datasets/download.py list-files smartspaces     # browse HF repo

Every finished file is recorded with its size and sha256 in <root>/DOWNLOADS.json
so the team can confirm everyone benchmarks on identical data.
Request-only datasets (MSMT17, MMPTRACK) print the access steps instead.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
UA = {"User-Agent": "rvb-downloader/1.0"}


def load_manifest() -> dict[str, dict]:
    data = json.loads(MANIFEST.read_text())
    return {d["id"]: d for d in data["datasets"]}


# ---------------------------------------------------------------- helpers
def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def record(root: Path, ds_id: str, path: Path, url: str) -> None:
    log_path = root / "DOWNLOADS.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    log.setdefault(ds_id, {})[str(path.relative_to(root))] = {
        "url": url, "bytes": path.stat().st_size, "sha256": sha256(path),
        "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log_path.write_text(json.dumps(log, indent=1))


def http_download(url: str, dest: Path, retries: int = 5) -> Path:
    """Resumable HTTP(S) download using Range requests; skips complete files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            head = urllib.request.Request(url, method="HEAD", headers=UA)
            try:
                with urllib.request.urlopen(head, timeout=60) as r:
                    total = int(r.headers.get("Content-Length") or 0)
            except urllib.error.HTTPError:
                total = 0
            if dest.exists() and (total == 0 or dest.stat().st_size == total):
                print(f"  = {dest.name} already complete")
                return dest
            have = part.stat().st_size if part.exists() else 0
            headers = dict(UA)
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                done, t0 = have, time.time()
                with open(part, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100 * done / total
                            rate = (done - have) / max(time.time() - t0, 1e-3) / 1e6
                            print(f"\r  {dest.name}: {pct:5.1f}%  {done/1e9:.2f}/{total/1e9:.2f} GB  {rate:.1f} MB/s", end="")
                print()
            if total and part.stat().st_size != total:
                raise IOError(f"size mismatch {part.stat().st_size} != {total}")
            part.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            print(f"\n  ! attempt {attempt}/{retries} failed: {e}")
            time.sleep(min(60, 5 * attempt))
    raise SystemExit(f"FAILED: {url}")


def unzip(path: Path, into: Path) -> None:
    print(f"  unzipping {path.name} -> {into}")
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"corrupt zip member {bad} in {path}")
        z.extractall(into)


def list_index(url: str) -> list[str]:
    """Return hrefs from a plain Apache-style directory index."""
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    hrefs = re.findall(r'href="([^"?#]+)"', html)
    return sorted({h for h in hrefs if not h.startswith(("/", "..", "http", "mailto"))})


def need(cmd: str, hint: str) -> None:
    if shutil.which(cmd) is None:
        raise SystemExit(f"'{cmd}' not found. {hint}")


# ---------------------------------------------------------------- per-access handlers
def get_direct(ds, root, args):
    out = root / ds["id"]
    for f in ds["files"]:
        if f.get("optional") and not args.all_files:
            continue
        name = Path(urllib.parse.urlparse(f["url"]).path).name
        dest = http_download(f["url"], out / "_archives" / name)
        record(root, ds["id"], dest, f["url"])
        if f.get("unzip"):
            unzip(dest, out)
    if ds.get("mirrors"):
        print("  mirrors if the main link fails:", *ds["mirrors"], sep="\n    ")


def get_caviar(ds, root, args):
    out = root / "caviar"
    for seq in ds["sequences"]:
        url = ds["base_url"] + seq + "/"
        try:
            names = list_index(url)
        except Exception as e:  # noqa: BLE001
            print(f"  ! cannot list {url}: {e}")
            continue
        exts = (".mpg", ".xml") if args.with_videos or True else (".xml",)
        wanted = [n for n in names if n.lower().endswith(exts)]  # whole set is ~0.4 GB
        for n in wanted:
            dest = http_download(url + n, out / seq / Path(n).name)
            record(root, "caviar", dest, url + n)


def get_index(ds, root, args):
    out = root / ds["id"]
    stack = [ds["base_url"]]
    while stack:
        url = stack.pop()
        for n in list_index(url):
            if n.endswith("/"):
                stack.append(url + n)
            else:
                rel = (url + n)[len(ds["base_url"]):]
                dest = http_download(url + n, out / rel)
                record(root, ds["id"], dest, url + n)


def get_huggingface(ds, root, args):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub  (then `huggingface-cli login` if the repo is gated)")
    include = list(ds.get("default_include", []))
    if args.with_videos:
        include += ds.get("video_include", [])
    if args.include:
        include = args.include
    print(f"  include patterns: {include}\n  exclude patterns: {ds.get('exclude')}")
    path = snapshot_download(
        repo_id=ds["repo_id"], repo_type=ds.get("repo_type", "dataset"),
        local_dir=str(root / ds["id"]), allow_patterns=include or None,
        ignore_patterns=ds.get("exclude") or None, max_workers=8,
    )
    print(f"  saved under {path}")


def list_hf_files(ds, pattern: str | None):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(ds["repo_id"], repo_type=ds.get("repo_type", "dataset"))
    scenes = sorted({"/".join(f.split("/")[:3]) for f in files if f.count("/") >= 3})
    for s in scenes:
        if not pattern or fnmatch.fnmatch(s, pattern):
            print(s)


def get_s3(ds, root, args):
    need("aws", "pip install awscli")
    src = ds["bucket"].rstrip("/") + "/" + (args.prefix or ds.get("default_prefix", ""))
    if not args.prefix:
        print(f"  listing {src} (pass --prefix <folder>/ to download one folder)")
        subprocess.run(["aws", "s3", "ls", "--no-sign-request", src], check=True)
        return
    dest = root / ds["id"] / args.prefix
    cmd = ["aws", "s3", "sync", "--no-sign-request", src, str(dest)]
    if args.dry_run:
        cmd.append("--dryrun")
    subprocess.run(cmd, check=True)


def get_repo_script(ds, root, args):
    need("git", "install git")
    need("ffmpeg", "install ffmpeg (apt install ffmpeg)")
    need("aws", "pip install awscli")
    repo_dir = root / ds["id"] / Path(ds["repo"]).stem
    if not repo_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", ds["repo"], str(repo_dir)], check=True)
    if ds.get("requirements"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", ds["requirements"]], cwd=repo_dir, check=True)
    subprocess.run(ds["script"].split(), cwd=repo_dir, check=True)


def get_request(ds, root, args):
    print(f"""  {ds['name']} needs a signed request - it cannot be scripted.
    1. Go to {ds['request_url']}
    2. Agreement: {ds['agreement']}
    3. When you get the link, download and unzip into: {root / ds['id']}
    4. Run: python datasets/download.py verify --root {root}
  Licence: {ds['licence']}""")


HANDLERS = {"direct": get_direct, "caviar": get_caviar, "index": get_index, "huggingface": get_huggingface,
            "s3": get_s3, "repo_script": get_repo_script, "request": get_request}


# ---------------------------------------------------------------- commands
def cmd_list(m, args):
    print(f"{'id':14} {'pri':3} {'tasks':28} {'size':>7} {'commercial':10} access")
    for d in m.values():
        size = d.get("size_gb") or sum(f["size_gb"] for f in d.get("files", []))
        print(f"{d['id']:14} {d['priority']:3} {','.join(d['tasks']):28} {size:6.1f}G "
              f"{'yes' if d['commercial_ok'] else 'no':10} {d['access']}")
    print("\nLicences: see manifest.json. Only smartspaces and meva allow commercial use.")


def cmd_get(m, args):
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    ids = [d for d in m if m[d]["priority"] == "P0"] if args.ids == ["p0"] else args.ids
    for ds_id in ids:
        if ds_id not in m:
            raise SystemExit(f"unknown dataset '{ds_id}'. Try: {', '.join(m)}")
        ds = m[ds_id]
        print(f"\n== {ds['name']}  [{ds['licence']}]")
        free = shutil.disk_usage(root).free / 1e9
        size = ds.get("size_gb") or sum(f["size_gb"] for f in ds.get("files", []))
        if ds["access"] not in ("request", "s3") and size * 2.2 > free:
            raise SystemExit(f"  only {free:.0f} GB free; {ds_id} needs ~{size*2.2:.0f} GB incl. unzip")
        HANDLERS[ds["access"]](ds, root, args)


def cmd_verify(m, args):
    root = Path(args.root).expanduser().resolve()
    ok = True
    for d in m.values():
        exp = d.get("expect", [])
        if not exp:
            continue
        present = [root / e if (e == d["id"] or e.startswith(d["id"] + "/")) else root / d["id"] / e for e in exp]
        status = all(p.exists() for p in present)
        ok &= status or d["priority"] != "P0"
        print(f"{'OK  ' if status else 'MISS'} {d['id']:14} {d['priority']}  " + ("" if status else f"missing: {[str(p) for p in present if not p.exists()]}"))
    mot = root / "mot17" / "MOT17" / "train"
    if mot.exists():
        seqs = [p.name for p in mot.iterdir() if (p / "gt" / "gt.txt").exists()]
        print(f"     mot17: {len(seqs)} train sequences with GT")
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("get")
    g.add_argument("ids", nargs="+", help="dataset ids, or 'p0' for all P0 datasets")
    g.add_argument("--root", default=os.environ.get("RVB_DATA", "./data"))
    g.add_argument("--with-videos", action="store_true", help="HF datasets: also pull videos")
    g.add_argument("--include", nargs="*", help="HF: override include patterns")
    g.add_argument("--prefix", help="S3: folder to sync")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--all-files", action="store_true", help="also fetch optional files")
    v = sub.add_parser("verify")
    v.add_argument("--root", default=os.environ.get("RVB_DATA", "./data"))
    lf = sub.add_parser("list-files")
    lf.add_argument("id")
    lf.add_argument("--pattern")
    args = p.parse_args()
    m = load_manifest()
    if args.cmd == "list":
        cmd_list(m, args)
    elif args.cmd == "get":
        cmd_get(m, args)
    elif args.cmd == "verify":
        cmd_verify(m, args)
    elif args.cmd == "list-files":
        list_hf_files(m[args.id], args.pattern)


if __name__ == "__main__":
    main()
