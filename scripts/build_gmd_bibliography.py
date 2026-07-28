#!/usr/bin/env python3
"""Build the manuscript bibliography from verified Crossref candidates."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


SURVEY_OVERRIDES = {
    "survey23": 2,  # final JAMES article, not the ESSOAr posting
    "survey26": 1,  # final npj article, not the Research Square posting
}


MANUAL = r"""
@article{Ascenso2024,
  author = {Ascenso, Guido and Palcic, Giulio and Scoccimarro, Enrico and Castelletti, Andrea},
  title = {A systematic framework for data augmentation for tropical cyclone intensity estimation using deep learning},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  year = {2024},
  volume = {1},
  number = {3},
  pages = {e2024JH000206},
  doi = {10.1029/2024JH000206}
}

@article{Ascenso2026,
  author = {Ascenso, Guido and Giuliani, Matteo and P{\'e}rez-Aracil, Jorge and Salcedo-Sanz, Sancho and Bertini, Claudia and Bonetti, Paolo and Camps-Valls, Gustau and Fern{\'a}ndez Torres, Miguel {\'A}ngel and Ficch{\`i}, Andrea and Linscheid, Nora and Merlo, Martina and Palcic, Giulio and Reichstein, Markus and Restelli, Marcello and Toreti, Andrea and Walt, Eliot and Castelletti, Andrea},
  title = {The maturation of {AI} in drought science: A review of trends, pitfalls, and priorities},
  journal = {Water Resources Research},
  year = {2026},
  volume = {62},
  number = {4},
  pages = {e2025WR041828},
  doi = {10.1029/2025WR041828}
}

@inproceedings{Heusel2017,
  author = {Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
  title = {GANs trained by a two time-scale update rule converge to a local Nash equilibrium},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2017},
  volume = {30},
  pages = {6626--6637},
  url = {https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html}
}

@inproceedings{Chong2020,
  author = {Chong, Min Jin and Forsyth, David},
  title = {Effectively unbiased {FID} and Inception Score and where to find them},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2020},
  pages = {6069--6078},
  doi = {10.1109/CVPR42600.2020.00611}
}

@inproceedings{Kynkaanniemi2023,
  author = {Kynk{\"a}{\"a}nniemi, Tuomas and Karras, Tero and Aittala, Miika and Aila, Timo and Lehtinen, Jaakko},
  title = {The role of {ImageNet} classes in {Fr{\'e}chet} Inception Distance},
  booktitle = {International Conference on Learning Representations},
  year = {2023},
  url = {https://openreview.net/forum?id=4oXTQ6m_ws8}
}

@inproceedings{Parmar2022,
  author = {Parmar, Gaurav and Zhang, Richard and Zhu, Jun-Yan},
  title = {On aliased resizing and surprising subtleties in {GAN} evaluation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2022},
  pages = {11400--11410},
  doi = {10.1109/CVPR52688.2022.01112}
}

@inproceedings{Jayasumana2024,
  author = {Jayasumana, Sadeep and Ramalingam, Srikumar and Veit, Andreas and Glasner, Daniel and Chakrabarti, Ayan and Kumar, Sanjiv},
  title = {Rethinking {FID}: Towards a better evaluation metric for image generation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2024},
  pages = {9307--9315},
  doi = {10.1109/CVPR52733.2024.00889}
}

@article{Seibert2018,
  author = {Seibert, Jan and Vis, Marc J. P. and Lewis, Elizabeth and van Meerveld, H. J.},
  title = {Upper and lower benchmarks in hydrological modelling},
  journal = {Hydrological Processes},
  year = {2018},
  volume = {32},
  number = {8},
  pages = {1120--1125},
  doi = {10.1002/hyp.11476}
}

@article{Gretton2012,
  author = {Gretton, Arthur and Borgwardt, Karsten M. and Rasch, Malte J. and Sch{\"o}lkopf, Bernhard and Smola, Alexander},
  title = {A kernel two-sample test},
  journal = {Journal of Machine Learning Research},
  year = {2012},
  volume = {13},
  pages = {723--773},
  url = {https://www.jmlr.org/papers/v13/gretton12a.html}
}

@inproceedings{Kynkaanniemi2019,
  author = {Kynk{\"a}{\"a}nniemi, Tuomas and Karras, Tero and Laine, Samuli and Lehtinen, Jaakko and Aila, Timo},
  title = {Improved precision and recall metric for assessing generative models},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2019},
  volume = {32},
  url = {https://proceedings.neurips.cc/paper/2019/hash/0234c510bc6d908b28c70ff313743079-Abstract.html}
}

@inproceedings{Naeem2020,
  author = {Naeem, Muhammad Ferjad and Oh, Seong Joon and Uh, Youngjung and Choi, Yunjey and Yoo, Jaejun},
  title = {Reliable fidelity and diversity metrics for generative models},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  year = {2020},
  volume = {119},
  pages = {7176--7185},
  series = {Proceedings of Machine Learning Research},
  url = {https://proceedings.mlr.press/v119/naeem20a.html}
}

@inproceedings{Carlini2023,
  author = {Carlini, Nicholas and Hayes, Jamie and Nasr, Milad and Jagielski, Matthew and Sehwag, Vikash and Tram{\`e}r, Florian and Balle, Borja and Ippolito, Daphne and Wallace, Eric},
  title = {Extracting training data from diffusion models},
  booktitle = {32nd USENIX Security Symposium},
  year = {2023},
  pages = {5253--5270},
  url = {https://www.usenix.org/conference/usenixsecurity23/presentation/carlini}
}

@inproceedings{Somepalli2023,
  author = {Somepalli, Gowthami and Singla, Vasu and Goldblum, Micah and Geiping, Jonas and Goldstein, Tom},
  title = {Diffusion art or digital forgery? Investigating data replication in diffusion models},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2023},
  pages = {6048--6058},
  doi = {10.1109/CVPR52729.2023.00586}
}

@article{Knapp2010,
  author = {Knapp, Kenneth R. and Kruk, Michael C. and Levinson, David H. and Diamond, Howard J. and Neumann, Charles J.},
  title = {The International Best Track Archive for Climate Stewardship ({IBTrACS}): Unifying tropical cyclone data},
  journal = {Bulletin of the American Meteorological Society},
  year = {2010},
  volume = {91},
  number = {3},
  pages = {363--376},
  doi = {10.1175/2009BAMS2755.1}
}

@article{Knapp2011,
  author = {Knapp, Kenneth R. and Ansari, Steve and Bain, Caroline L. and Bourassa, Mark A. and Dickinson, Michael J. and Funk, Chris and Helms, Chip N. and Hennon, Christopher C. and Holmes, C. D. and Huffman, George J. and Kossin, James P. and Lee, Hai-Tien and Loew, Alexander and Magnusdottir, Gudrun},
  title = {Globally gridded satellite observations for climate studies},
  journal = {Bulletin of the American Meteorological Society},
  year = {2011},
  volume = {92},
  number = {7},
  pages = {893--907},
  doi = {10.1175/2011BAMS3039.1}
}

@article{Pineros2008,
  author = {Pi{\~n}eros, Miguel F. and Ritchie, Elizabeth A. and Tyo, J. Scott},
  title = {Objective measures of tropical cyclone structure and intensity change from remotely sensed infrared image data},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year = {2008},
  volume = {46},
  number = {11},
  pages = {3574--3580},
  doi = {10.1109/TGRS.2008.2000819}
}

@article{Ritchie2012,
  author = {Ritchie, Elizabeth A. and Valliere-Kelley, Genevieve and Pi{\~n}eros, Miguel F. and Tyo, J. Scott},
  title = {Tropical cyclone intensity estimation in the North Atlantic basin using an improved deviation angle variance technique},
  journal = {Weather and Forecasting},
  year = {2012},
  volume = {27},
  number = {5},
  pages = {1264--1277},
  doi = {10.1175/WAF-D-11-00156.1}
}

@book{Villani2009,
  author = {Villani, C{\'e}dric},
  title = {Optimal Transport: Old and New},
  publisher = {Springer},
  address = {Berlin, Heidelberg},
  year = {2009},
  doi = {10.1007/978-3-540-71050-9}
}

@book{DavisonHinkley1997,
  author = {Davison, Anthony C. and Hinkley, David V.},
  title = {Bootstrap Methods and Their Application},
  publisher = {Cambridge University Press},
  address = {Cambridge},
  year = {1997},
  doi = {10.1017/CBO9780511802843}
}

@article{CameronMiller2015,
  author = {Cameron, A. Colin and Miller, Douglas L.},
  title = {A practitioner's guide to cluster-robust inference},
  journal = {Journal of Human Resources},
  year = {2015},
  volume = {50},
  number = {2},
  pages = {317--372},
  doi = {10.3368/jhr.50.2.317}
}

@article{Holm1979,
  author = {Holm, Sture},
  title = {A simple sequentially rejective multiple test procedure},
  journal = {Scandinavian Journal of Statistics},
  year = {1979},
  volume = {6},
  number = {2},
  pages = {65--70},
  url = {https://www.jstor.org/stable/4615733}
}

@article{Dunn1961,
  author = {Dunn, Olive Jean},
  title = {Multiple comparisons among means},
  journal = {Journal of the American Statistical Association},
  year = {1961},
  volume = {56},
  number = {293},
  pages = {52--64},
  doi = {10.1080/01621459.1961.10482090}
}

@article{Wilson1927,
  author = {Wilson, Edwin B.},
  title = {Probable inference, the law of succession, and statistical inference},
  journal = {Journal of the American Statistical Association},
  year = {1927},
  volume = {22},
  number = {158},
  pages = {209--212},
  doi = {10.1080/01621459.1927.10502953}
}

@article{GneitingRaftery2007,
  author = {Gneiting, Tilmann and Raftery, Adrian E.},
  title = {Strictly proper scoring rules, prediction, and estimation},
  journal = {Journal of the American Statistical Association},
  year = {2007},
  volume = {102},
  number = {477},
  pages = {359--378},
  doi = {10.1198/016214506000001437}
}

@inproceedings{Ho2020,
  author = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  title = {Denoising diffusion probabilistic models},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
  volume = {33},
  pages = {6840--6851},
  url = {https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html}
}

@inproceedings{Karras2022,
  author = {Karras, Tero and Aittala, Miika and Aila, Timo and Laine, Samuli},
  title = {Elucidating the design space of diffusion-based generative models},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2022},
  volume = {35},
  pages = {26565--26577},
  url = {https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html}
}

@inproceedings{Nichol2021,
  author = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
  title = {Improved denoising diffusion probabilistic models},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  year = {2021},
  volume = {139},
  series = {Proceedings of Machine Learning Research},
  pages = {8162--8171},
  url = {https://proceedings.mlr.press/v139/nichol21a.html}
}

@inproceedings{Lin2024,
  author = {Lin, Shanchuan and Liu, Bingchen and Li, Jiashi and Yang, Xiao},
  title = {Common diffusion noise schedules and sample steps are flawed},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year = {2024},
  pages = {5404--5411},
  url = {https://openaccess.thecvf.com/content/WACV2024/html/Lin_Common_Diffusion_Noise_Schedules_and_Sample_Steps_Are_Flawed_WACV_2024_paper.html}
}

@inproceedings{Hang2023,
  author = {Hang, Tiankai and Gu, Shuyang and Li, Chen and Bao, Jianmin and Chen, Dong and Hu, Han and Geng, Xin and Guo, Baining},
  title = {Efficient diffusion training via min-SNR weighting strategy},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year = {2023},
  pages = {7441--7451},
  url = {https://openaccess.thecvf.com/content/ICCV2023/html/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.html}
}

@inproceedings{Saharia2022,
  author = {Saharia, Chitwan and Chan, William and Saxena, Saurabh and Li, Lala and Whang, Jay and Denton, Emily L. and Ghasemipour, Kamyar and Ayan, Burcu Karagol and Salimans, Tim and Ho, Jonathan and Fleet, David J. and Norouzi, Mohammad},
  title = {Photorealistic text-to-image diffusion models with deep language understanding},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2022},
  volume = {35},
  url = {https://proceedings.neurips.cc/paper_files/paper/2022/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html}
}

@inproceedings{Lu2023,
  author = {Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun},
  title = {DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models},
  booktitle = {International Conference on Learning Representations},
  year = {2023},
  url = {https://openreview.net/forum?id=4vGwQqviud5}
}

@article{Harris2020NumPy,
  author = {Harris, Charles R. and Millman, K. Jarrod and van der Walt, St{\'e}fan J. and others},
  title = {Array programming with {NumPy}},
  journal = {Nature},
  year = {2020},
  volume = {585},
  pages = {357--362},
  doi = {10.1038/s41586-020-2649-2}
}

@article{Virtanen2020SciPy,
  author = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and others},
  title = {{SciPy} 1.0: fundamental algorithms for scientific computing in {Python}},
  journal = {Nature Methods},
  year = {2020},
  volume = {17},
  pages = {261--272},
  doi = {10.1038/s41592-019-0686-2}
}

@article{Hunter2007,
  author = {Hunter, John D.},
  title = {Matplotlib: A 2D graphics environment},
  journal = {Computing in Science \& Engineering},
  year = {2007},
  volume = {9},
  number = {3},
  pages = {90--95},
  doi = {10.1109/MCSE.2007.55}
}

@misc{NHCScale,
  author = {{National Hurricane Center}},
  title = {Saffir--Simpson Hurricane Wind Scale},
  year = {2026},
  url = {https://www.nhc.noaa.gov/aboutsshws.php},
  note = {Last access: 23 July 2026}
}
"""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("manuscript/gmd/reference_candidates.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manuscript/gmd/references.bib"),
    )
    return parser.parse_args()


def _latex(value: str) -> str:
    value = html.unescape(value)
    replacements = {
        "ß": r"{\ss}",
        "á": r"{\'a}",
        "é": r"{\'e}",
        "í": r"{\'i}",
        "î": r"{\^\i}",
        "ò": r"{\`o}",
        "ö": r"{\"o}",
        "ü": r"{\"u}",
        "Š": r"{\v S}",
        "š": r"{\v s}",
        "ů": r"{\r u}",
        "‐": "-",
        "‑": "-",
        "‒": "--",
        "–": "--",
        "—": "---",
        "−": "-",
        "’": "'",
        "“": "``",
        "”": "''",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = value.replace("&", r"\&")
    value = value.replace("%", r"\%")
    value = value.replace("_", r"\_")
    value = value.replace("#", r"\#")
    return re.sub(r"\s+", " ", value).strip()


def _authors(candidate: dict) -> str:
    return " and ".join(
        f"{_latex(author['family'])}, {_latex(author['given'])}" for author in candidate["authors"]
    )


def _entry(key: str, candidate: dict) -> str:
    kind = candidate["type"]
    if kind == "journal-article":
        entry_type = "article"
    elif kind in {"proceedings-article", "book-chapter"}:
        entry_type = "inproceedings"
    else:
        entry_type = "misc"
    fields = [
        ("author", _authors(candidate)),
        ("title", _latex(candidate["title"])),
        ("year", str(candidate["year"])),
    ]
    container = candidate.get("container_title")
    if container:
        fields.append(
            (
                "journal" if entry_type == "article" else "booktitle",
                _latex(container),
            )
        )
    for field in ("volume", "issue", "page", "publisher", "doi"):
        value = candidate.get(field)
        if not value:
            continue
        bib_field = {"issue": "number", "page": "pages"}.get(field, field)
        fields.append(
            (
                bib_field,
                _latex(str(value)).replace("-", "--") if field == "page" else _latex(str(value)),
            )
        )
    if not candidate.get("doi") and candidate.get("url"):
        fields.append(("url", _latex(candidate["url"])))
    body = ",\n".join(f"  {field} = {{{value}}}" for field, value in fields)
    return f"@{entry_type}{{{key},\n{body}\n}}\n"


def main() -> None:
    args = _arguments()
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    entries = [MANUAL.strip(), ""]
    for index in range(1, 70):
        key = f"survey{index:02d}"
        result = payload["results"][key]
        candidate_index = SURVEY_OVERRIDES.get(key, 0)
        candidate = result["candidates"][candidate_index]
        entries.append(_entry(key, candidate))
    args.output.write_text("\n".join(entries).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
