#!/usr/bin/env python3
"""Query Crossref for bibliography candidates and retain auditable match scores."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


EXTRA_TITLES = {
    "heusel2017fid": "GANs trained by a two time-scale update rule converge to a local Nash equilibrium",
    "gretton2012mmd": "A kernel two-sample test",
    "kynkaanniemi2019pr": "Improved precision and recall metric for assessing generative models",
    "naeem2020dc": "Reliable fidelity and diversity metrics for generative models",
    "carlini2023diffusion": "Extracting training data from diffusion models",
    "somepalli2023copies": "Diffusion art or digital forgery? Investigating data replication in diffusion models",
    "efron1994bootstrap": "An introduction to the bootstrap",
    "knapp2010ibtracs": "The International Best Track Archive for Climate Stewardship (IBTrACS)",
    "knapp2011gridsat": "Globally gridded satellite observations for climate studies",
    "page2021prisma": "The PRISMA 2020 statement: an updated guideline for reporting systematic reviews",
    "bonferroni1936": "Teoria statistica delle classi e calcolo delle probabilita",
}


def _normalized(value: str) -> str:
    return " ".join(
        "".join(character.lower() if character.isalnum() else " " for character in value).split()
    )


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return default


def _year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        parts = item.get(key, {}).get("date-parts", [])
        if parts and parts[0] and parts[0][0] is not None:
            return int(parts[0][0])
    return None


def _candidate(
    item: dict[str, Any], *, query_title: str, expected_year: int | None
) -> dict[str, Any]:
    title = str(_first(item.get("title"), ""))
    similarity = SequenceMatcher(None, _normalized(query_title), _normalized(title)).ratio()
    candidate_year = _year(item)
    year_penalty = (
        0.0
        if expected_year is None or candidate_year is None
        else min(abs(candidate_year - expected_year) * 0.03, 0.15)
    )
    return {
        "match_score": round(similarity - year_penalty, 6),
        "title_similarity": round(similarity, 6),
        "query_title": query_title,
        "title": title,
        "year": candidate_year,
        "doi": item.get("DOI"),
        "url": item.get("URL"),
        "type": item.get("type"),
        "container_title": _first(item.get("container-title"), ""),
        "volume": item.get("volume"),
        "issue": item.get("issue"),
        "page": item.get("page") or item.get("article-number"),
        "publisher": item.get("publisher"),
        "authors": [
            {
                "given": author.get("given", ""),
                "family": author.get("family", ""),
                "orcid": author.get("ORCID"),
            }
            for author in item.get("author", [])
        ],
    }


def _query(title: str, *, expected_year: int | None, rows: int) -> list[dict[str, Any]]:
    parameters = urllib.parse.urlencode(
        {
            "query.title": title,
            "rows": rows,
            "select": (
                "DOI,title,author,published-print,published-online,issued,created,"
                "container-title,volume,issue,page,article-number,publisher,type,URL"
            ),
        }
    )
    request = urllib.request.Request(
        f"https://api.crossref.org/works?{parameters}",
        headers={"User-Agent": "GAUGE-reference-audit/0.1"},
    )
    payload = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(2.0 ** (attempt + 1))
    if payload is None:
        raise RuntimeError("Crossref returned no payload.")
    candidates = [
        _candidate(item, query_title=title, expected_year=expected_year)
        for item in payload["message"]["items"]
    ]
    return sorted(candidates, key=lambda item: item["match_score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--survey",
        type=Path,
        default=Path("survey/exports/coded_papers.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/gmd/reference_candidates.json"),
    )
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.12)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retain successful queries already present in the output and retry only failures.",
    )
    parser.add_argument(
        "--keys",
        help="Optional comma-separated subset of query keys.",
    )
    args = parser.parse_args()

    queries: list[tuple[str, str, int | None, str]] = []
    with args.survey.open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            queries.append(
                (
                    f"survey{index:02d}",
                    row["Title"],
                    int(float(row["Year"])),
                    "survey",
                )
            )
    queries.extend((key, title, None, "foundation") for key, title in EXTRA_TITLES.items())

    previous = {}
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8")).get("results", {})
    requested_keys = (
        None
        if not args.keys
        else {value.strip() for value in args.keys.split(",") if value.strip()}
    )
    results = dict(previous)
    for number, (key, title, expected_year, role) in enumerate(queries, start=1):
        if requested_keys is not None and key not in requested_keys:
            continue
        if key in previous and previous[key].get("error") is None:
            continue
        try:
            candidates = _query(title, expected_year=expected_year, rows=args.rows)
            error = None
        except Exception as exc:  # noqa: BLE001 - retain per-query audit failures
            candidates = []
            error = f"{type(exc).__name__}: {exc}"
        results[key] = {
            "role": role,
            "expected_title": title,
            "expected_year": expected_year,
            "candidates": candidates,
            "error": error,
        }
        print(
            f"{number:03d}/{len(queries):03d} {key}: {candidates[0]['match_score'] if candidates else error}"
        )
        time.sleep(max(args.delay, 0.0))

    payload = {
        "schema": "gauge_climate.reference_candidates.v1",
        "source": str(args.survey),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
