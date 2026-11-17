"""Fetch pinned public source files for inspection; never import or execute them."""

import base64
import concurrent.futures
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SPECS = [
    ("yaozhwen/wechatOcr", "b2cde637550e1406f15887cd7bc32e6f6c5716c2"),
    ("yuanzal/Weclip", "d3aa61d521b36ae9ad6081e86c163aa1cf6ecd7f"),
]
BASE = Path(__file__).resolve().parent


def request_json(url):
    last = None
    for _ in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "WeLog-source-review"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.load(response)
        except Exception as error:
            last = error
    raise last


def git_hash(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main():
    snapshots, jobs = [], []
    for repo, commit in SPECS:
        tree = request_json(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
        assert not tree.get("truncated")
        directory = BASE / "sources" / (repo.replace("/", "-") + "-" + commit[:12])
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = {"repo": repo, "commit": commit, "directory": directory.relative_to(BASE.parent).as_posix(), "files": []}
        snapshots.append(snapshot)
        for entry in tree["tree"]:
            if entry["type"] != "blob" or entry["mode"] != "100644" or entry.get("size", 0) > 500000:
                continue
            if Path(entry["path"]).name in {"AGENTS.md", "SKILL.md"}:
                continue
            target = (directory / entry["path"]).resolve()
            assert target.is_relative_to(directory.resolve())
            jobs.append((snapshot, entry, target))

    def fetch(job):
        snapshot, entry, target = job
        data = target.read_bytes() if target.exists() else b""
        if git_hash(data) != entry["sha"]:
            blob = request_json(f"https://api.github.com/repos/{snapshot['repo']}/git/blobs/{entry['sha']}")
            data = base64.b64decode(blob["content"])
            assert git_hash(data) == entry["sha"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return snapshot, {"path": entry["path"], "bytes": len(data), "git_blob": entry["sha"],
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "url": f"https://github.com/{snapshot['repo']}/blob/{snapshot['commit']}/{entry['path']}"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for snapshot, file in pool.map(fetch, jobs):
            snapshot["files"].append(file)
    for snapshot in snapshots:
        snapshot["files"].sort(key=lambda file: file["path"])
    (BASE / "source-manifest.json").write_text(json.dumps({
        "downloaded_at": datetime.now(timezone.utc).isoformat(), "snapshots": snapshots,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{"repo": item["repo"], "commit": item["commit"], "files": len(item["files"])}
                      for item in snapshots], indent=2))


if __name__ == "__main__":
    main()
