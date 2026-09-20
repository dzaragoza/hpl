"""Download TOP500 history from top500.org into top500_data.json.

Run this once (with network access) to (re)generate the data file that
hpl_np.py uses for its "when would this machine have been a
supercomputer?" calibration.  hpl_np.py itself never touches the
network; it only reads the JSON this script writes.

Each edition of the TOP500 list (June 1993 through the latest) is
published at:

    https://www.top500.org/lists/top500/YYYY/MM/download/TOP500_YYYYMM_all.xml

We keep only what the calibration needs per edition: the #1 system's
Rmax (would your machine have topped the list?) and the #500 system's
Rmax (would it have made the list at all?), plus a label.  All 67
editions compress to a few kilobytes.

Usage:

    python3 top500_update.py            # writes top500_data.json here
"""

import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

NS = "{http://www.top500.org/xml/top500/1.0}"
DATA_FILE = "top500_data.json"
UA = "Mozilla/5.0 (X11; Linux x86_64) hpl-py-educational/1.0"


def edition_months(start_year, end_year):
    """Candidate (year, month) pairs.  Editions are June/November, with
    one historical exception (December 1995); 404s are skipped, so
    probing 06/11/12 covers everything that exists."""
    months = []
    for year in range(start_year, end_year + 1):
        for month in (6, 11, 12):
            months.append((year, month))
    return months


def fetch_edition(year, month):
    """Return the parsed XML root for one edition, or None if absent."""
    ym = "{:04d}{:02d}".format(year, month)
    url = ("https://www.top500.org/lists/top500/{}/{:02d}/download/"
           "TOP500_{}_all.xml").format(year, month, ym)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        time.sleep(1.0)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return ET.fromstring(resp.read())
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise
    except (urllib.error.URLError, TimeoutError):
        for attempt in (2, 3):
            time.sleep(attempt * 5)
            try:
                with urllib.request.urlopen(req, timeout=90) as resp:
                    return ET.fromstring(resp.read())
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    return None
                raise
            except (urllib.error.URLError, TimeoutError):
                continue
        return None


def summarize_edition(root):
    """Extract (rmax of #1, rmax of #500, name of #1) from one list."""
    rmax_by_rank = {}
    top_name = None
    for site in root.iter(NS + "site"):
        rank = site.findtext(NS + "rank")
        rmax = site.findtext(NS + "r-max")
        if rank is None or rmax is None:
            continue
        try:
            rank = int(rank)
            rmax = float(rmax)
        except ValueError:
            continue
        rmax_by_rank[rank] = rmax
        if rank == 1:
            computer = site.findtext(NS + "computer") or ""
            maker = site.findtext(NS + "manufacturer") or ""
            top_name = "{}, {}".format(computer, maker)
    if 1 not in rmax_by_rank:
        return None
    entry = rmax_by_rank.get(500)
    if entry is None:
        entry = min(rmax_by_rank.values())
    return {
        "rmax_top": rmax_by_rank[1],
        "rmax_entry": entry,
        "top_system": top_name,
    }


def main():
    this_year = int(os.environ.get("TOP500_END_YEAR",
                                   __import__("time").strftime("%Y")))
    editions = []
    for year, month in edition_months(1993, this_year):
        root = fetch_edition(year, month)
        if root is None:
            continue
        summary = summarize_edition(root)
        if summary is None:
            print("warning: {}-{:02d} parsed but empty; skipped"
                  .format(year, month), file=sys.stderr)
            continue
        summary["edition"] = "{:04d}-{:02d}".format(year, month)
        summary["label"] = "{} {}".format(
            {6: "June", 11: "November", 12: "December"}[month], year)
        editions.append(summary)
        print("{:10s} #1: {:12.2f} Gflop/s   entry: {:12.2f} Gflop/s   {}"
              .format(summary["edition"], summary["rmax_top"],
                      summary["rmax_entry"], summary["top_system"]))

    editions.sort(key=lambda e: e["edition"])
    lines = ["{", ' "source": "https://www.top500.org (all TOP500 list '
            'editions, June 1993 - {})",'.format(editions[-1]["label"]),
             ' "editions": [']
    for i, ed in enumerate(editions):
        lines.append("  " + json.dumps(ed) +
                     ("," if i < len(editions) - 1 else ""))
    lines += [" ]", "}"]
    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\nwrote {} ({} editions, {} bytes)".format(
        DATA_FILE, len(editions), os.path.getsize(DATA_FILE)))
    return 0


if __name__ == "__main__":
    sys.exit(main())