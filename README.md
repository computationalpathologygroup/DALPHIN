# DALPHIN: a multicentric open benchmark for pathology AI copilots

> Vision-language models (VLMs) are rapidly emerging as interactive visual question answering (VQA) systems in digital pathology. Despite growing interest in clinical adoption, their ability to support pathologists as AI copilots for diagnostic tasks remains poorly understood. Independent, long-term benchmarking is essential to rigorously assess the clinical potential, robustness, and limitations of these AI copilots on diagnostically meaningful tasks. To enable fair evaluation and comparison of pathology AI copilots, we introduce the digital pathology AI copilot benchmark (DALPHIN), a multicentric open VQA benchmark for pathology AI copilots. DALPHIN consists of 300 cases collected across six healthcare institutions in six countries, covering 130 diagnoses from 14 pathology subspecialties, including non-neoplastic entities and rare cancers. The benchmark comprises 1,236 histopathology images (low-resolution whole-slide images and higher-resolution regions of interest) and 1,757 questions across six tasks: tissue/organ recognition, neoplastic status, neoplastic behavior (benign, malignant, in situ, or uncertain), diagnosis, and case-specific multiple-choice and free-response questions. The images and questions are publicly available on [Zenodo](https://zenodo.org/records/18609450), while the ground truth reference labels are sequestered and only used for automatic performance evaluation on [Grand Challenge](https://dalphin.grand-challenge.org/) to preserve the benchmark's integrity.

### Repository layout

Welcome to the GitHub repository for the DALPHIN benchmark. This repository provides:

* Code to download the dataset from the associated [Zenodo repository](https://zenodo.org/records/18609450)
* A reference implementation for generating answers on DALPHIN using Vision-Language Model (VLMs)
* Evaluation code identical to that used on [Grand Challenge](https://dalphin.grand-challenge.org/) for scoring model submissions

The repository is laid out as follows:

* The [`data/`](data/) directory starts out empty and is populated with files after running the [`download_all.sh`](download_all.sh) shell script, which downloads and extracts the dataset files. After extraction, the folder is organized as follows:

```bash
.
└── data/
    ├── images/                 # benchmark images in PNG format
    └── dalphin_metadata.csv    # benchmark questions and associated metadata
```

* The [`code/answer_generation/`](code/answer_generation/) directory contains:
  * A README describing the two answer generation scenarios used in the benchmark
  * A synchronous reference implementation ([`generate_answers_sync.py`](code/answer_generation/generate_answers_sync.py)) for running a VLM on DALPHIN, designed to be easily adapted to any model
  * The original asynchronous script ([`generate_answers_async.py`](code/answer_generation/generate_answers_async.py)) used for the DALPHIN study
* The [`code/gc_evaluation/`](code/gc_evaluation/) directory contains Python code used to evaluate submissions for each task on Grand Challenge. Reference labels and the organ recognition taxonomy are intentionally excluded, so this code is provided solely to illustrate the submission processing and evaluation pipeline.

We describe additional details regarding the dataset on our [Zenodo data repository](https://zenodo.org/records/18609450).

### Quickstart guide
1. Download all data from Zenodo by running the `download_all.sh` shell script. All data is automatically organized in the directory layout as described above.
2. To run your VLM on DALPHIN, consult the [answer generation README](code/answer_generation/README.md) and adapt the `generate_answer` function in [`generate_answers_sync.py`](code/answer_generation/generate_answers_sync.py) to fit your model's API.

### Citation & license
This GitHub repository is released under the [Apache-2.0 license](LICENSE). The data of the DALPHIN benchmark is released under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

If you use this benchmark, please cite:
```
@misc{lems2026dalphin,
      title={DALPHIN: Benchmarking Digital Pathology AI Copilots Against Pathologists on an Open Multicentric Dataset}, 
      author={Carlijn Lems and Sander Moonemans and Natálie Klubíčková and Biagio Brattoli and Taebum Lee and Seokhwi Kim and Veronica Vilaplana and Laura Pons and Sapir Hochman and Mauricio Eduardo Suárez-Franck and Pedro Luis Fernandez and Julius Drachneris and Donatas Petroska and Renaldas Augulis and Arvydas Laurinavicius and Domingos Oliveira and Diana Montezuma and Anouk B. Bouwmeester and Dominique van Midden and Anne-Marie Vos and Shoko Vos and Jolique van Ipenburg and Maschenka Balkenhol and Koen Winkler and Iris Nagtegaal and Konnie Hebeda and Uta Flucke and Katrien Grünberg and Josef Skopal and Brinder S. Chohan and Jordi Temprana-Salvador and Enrico Munari and Luca Cima and Giulia Querzoli and Yosamin Gonzalez Belisario and Jaeike W. Faber and Geert J. L. H. van Leenders and Jan H. von der Thüsen and Lodewijk A. A. Brosens and Ronald R. de Krijger and Pieter Wesseling and Sandrine Florquin and Mateusz Maniewski and Adam Kowalewski and Robert Barna and Dina Tiniakos and Joan Lop Gros and Rogier Donders and Jake S. F. Maurits and Ming Yang Lu and Chengkuan Chen and Faisal Mahmood and Jeroen van der Laak and Nadieh Khalili and Frédérique Meeuwsen and Francesco Ciompi},
      year={2026},
      eprint={2605.03544},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.03544}, 
}
```
