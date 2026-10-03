"""Check that every repo / DOI / paper URL in data/models.json is reachable.

Run from the repo root:   python code/scripts/check_links.py
No extra packages needed (standard library only).
"""
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "models.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (proposal-link-checker)"}


def reachable(url):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status < 400, str(r.status)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):  # publisher blocks scripts; link is probably fine
            return None, f"HTTP {e.code}"
        return False, f"HTTP {e.code}"
    except Exception as e:  # network error, DNS, timeout, ...
        return False, type(e).__name__


def main():
    models = json.loads(DATA.read_text(encoding="utf-8"))["models"]
    failures = 0
    for m in models:
        checks = []
        repo = m["code"].get("repo")
        if repo:
            checks.append(("repo", f"https://github.com/{repo}"))
        doi = m["paper"].get("doi")
        if doi:
            checks.append(("doi", f"https://doi.org/{doi}"))
        url = m["paper"].get("url")
        if url and not (doi and url.endswith(doi)):
            checks.append(("paper", url))
        if not checks:
            print(f"-    {m['name']:<24} (nothing to check: no repo/doi/url)")
            continue
        for kind, link in checks:
            ok, info = reachable(link)
            failures += 1 if ok is False else 0
            tag = "OK  " if ok else ("WARN" if ok is None else "FAIL")
            print(f"{tag} {m['name']:<24} {kind:<5} {info:<10} {link}")
    print(f"\n{failures} link(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
