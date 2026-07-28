#!/usr/bin/env python3
"""Build a deterministic, minimal Overleaf archive for the GMD manuscript."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript" / "gmd"
OVERLEAF_OUTPUT = ROOT / "manuscript" / "GAUGE_v0.2_GMD_Overleaf.zip"
PRISM_OUTPUT = ROOT / "manuscript" / "GAUGE_v0.2_GMD_Prism.zip"
FIXED_TIMESTAMP = (2026, 7, 23, 0, 0, 0)
FONTAWESOME_BLOCK = r"""\RequirePackage{fontawesome5}
\newcommand*{\faicon}[1]{{\csname faicon@#1\endcsname}}
\input{fontawesomesymbols-generic.tex}
\DeclareRobustCommand\FAone{\fontencoding{U}\fontfamily{fontawesomeone}\selectfont}
\DeclareRobustCommand\FAtwo{\fontencoding{U}\fontfamily{fontawesometwo}\selectfont}
\DeclareRobustCommand\FAthree{\fontencoding{U}\fontfamily{fontawesomethree}\selectfont}
\input{fontawesomesymbols-pdftex.tex}
\renewcommand{\faHourglass}[1][]{\faicon{hourglass\if\relax\detokenize{#1}\relax\else-#1\fi}}
\renewcommand{\faBattery}[1][4]{\faicon{battery-#1}}"""
PRISM_FONTAWESOME_STUB = r"""\providecommand{\faRibbon}{}
\providecommand{\faStar}{}
\providecommand{\faHourglass}[1][]{}
\providecommand{\faBattery}[1][4]{}"""


def members() -> list[Path]:
    names = [
        "manuscript.tex",
        "supplement.tex",
        "references.bib",
        "copernicus.cls",
        "copernicus.cfg",
        "copernicus.bst",
        "pdfscreen.sty",
        "pdfscreencop.sty",
        "README.md",
        "short_summary.txt",
        "generated_artifacts_manifest.json",
    ]
    names.extend(f"fig{index:02d}.pdf" for index in range(1, 8))
    names.extend(f"figS{index:02d}.pdf" for index in range(1, 24))
    names.extend(
        [
            "tables/global_composition.tex",
            "tables/native_values.tex",
            "tables/partitions.tex",
            "tables/results_macros.tex",
            "tables/survey_counts.tex",
            "tables/vmax_support.tex",
        ]
    )
    paths = [SOURCE / name for name in names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Overleaf inputs: {missing}")
    return paths


def _payload(path: Path, *, prism_compatibility: bool) -> bytes:
    payload = path.read_bytes()
    if path.name != "copernicus.cls" or not prism_compatibility:
        return payload
    text = payload.decode("utf-8")
    if text.count(FONTAWESOME_BLOCK) != 1:
        raise RuntimeError("Unexpected Copernicus FontAwesome block; refusing to patch.")
    return text.replace(FONTAWESOME_BLOCK, PRISM_FONTAWESOME_STUB).encode("utf-8")


def _write(output: Path, *, prism_compatibility: bool) -> None:
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in members():
            relative = path.relative_to(SOURCE).as_posix()
            info = ZipInfo(relative, FIXED_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(
                info,
                _payload(path, prism_compatibility=prism_compatibility),
            )
    print(f"Wrote {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes)")


def main() -> None:
    _write(OVERLEAF_OUTPUT, prism_compatibility=False)
    _write(PRISM_OUTPUT, prism_compatibility=True)


if __name__ == "__main__":
    main()
