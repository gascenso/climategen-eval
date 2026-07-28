#!/usr/bin/env python3
"""Copy checksummed GAUGE results into the GMD manuscript staging directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "case_studies/tropical_cyclone_bt/paper_results"
GMD = ROOT / "manuscript/gmd"

MAPPING = {
    "fig01.pdf": PAPER / "main/figure_1_framework_overview.pdf",
    # fig02.pdf is generated from the user-owned literature-review workstream.
    "fig03.pdf": PAPER / "main/figure_2_statistical_validation.pdf",
    "fig04.pdf": PAPER / "main/figure_3_tc_examples.pdf",
    "fig05.pdf": PAPER / "main/figure_4_reference_gap_profile.pdf",
    "fig06.pdf": PAPER / "main/figure_5_tail_physical_distributions.pdf",
    "fig07.pdf": PAPER / "main/figure_6_tail_spatial_structure.pdf",
    "figS01.pdf": PAPER / "appendix/figure_A1_conditional_gap_matrix.pdf",
    "figS02.pdf": PAPER / "appendix/figure_A2_sensitivity_controls.pdf",
    "figS03.pdf": PAPER / "appendix/figure_A3_all_class_radial_profiles.pdf",
    "figS04.pdf": PAPER / "appendix/figure_A4_all_class_radial_psd.pdf",
    "figS05.pdf": PAPER / "appendix/figure_A5_generation_audits.pdf",
    "figS06.pdf": PAPER / "appendix/figure_A_reference_stability.pdf",
    "figS07.pdf": PAPER / "appendix/figure_A_continuous_wind_response.pdf",
    "figS08.pdf": PAPER / "appendix/figure_A_feature_space_robustness.pdf",
    "figS09.pdf": PAPER / "appendix/extended_diagnostics/track1_feature_diagnostics.pdf",
    "figS10.pdf": PAPER / "appendix/extended_diagnostics/track1_dependence.pdf",
    "figS11.pdf": PAPER / "appendix/extended_diagnostics/track1_memorization.pdf",
    "figS12.pdf": PAPER / "appendix/extended_diagnostics/track1_preprocessing_saturation.pdf",
    "figS13.pdf": PAPER / "appendix/functional_distributions/min_core_bt.pdf",
    "figS14.pdf": PAPER / "appendix/functional_distributions/cold_frac_220.pdf",
    "figS15.pdf": PAPER / "appendix/functional_distributions/r_maxgrad.pdf",
    "figS16.pdf": PAPER / "appendix/functional_distributions/axisym.pdf",
    "figS17.pdf": PAPER / "appendix/functional_distributions/dav.pdf",
    "figS18.pdf": PAPER / "appendix/functional_distributions/eye_contrast.pdf",
    "figS19.pdf": PAPER / "appendix/functional_distributions/p01_bt.pdf",
    "figS20.pdf": PAPER / "appendix/figure_A2a_physical_controls.pdf",
    "figS21.pdf": PAPER / "appendix/figure_A2b_spatial_controls.pdf",
    "figS22.pdf": PAPER / "appendix/figure_A2c_feature_controls.pdf",
    "figS23.pdf": PAPER / "appendix/figure_A2d_invariance_and_resolution_controls.pdf",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    missing = [source for source in MAPPING.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "The paper bundle is incomplete; missing generated artifacts: "
            + ", ".join(str(path.relative_to(ROOT)) for path in missing)
        )
    GMD.mkdir(parents=True, exist_ok=True)
    records = {}
    for destination_name, source in MAPPING.items():
        destination = GMD / destination_name
        shutil.copy2(source, destination)
        records[destination_name] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": _sha256(source),
            "staged_sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
        }
    survey_figure = GMD / "fig02.pdf"
    if not survey_figure.is_file():
        raise FileNotFoundError(
            "The structured-review figure is missing; run the user-owned survey "
            "artifact workflow before packaging."
        )
    records["fig02.pdf"] = {
        "source": "user-owned structured-review artifact",
        "staged_sha256": _sha256(survey_figure),
        "bytes": survey_figure.stat().st_size,
    }
    manifest = {
        "schema": "gauge_climate.gmd_generated_artifacts.v1",
        "artifacts": records,
    }
    manifest_path = GMD / "generated_artifacts_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "staging_directory": str(GMD),
                "artifact_count": len(records),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
