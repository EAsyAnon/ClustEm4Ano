# ClustEm4Ano
Implementation of complementary methods for data anonymization through generalization and suppression.
The repository contains code to generate value generalization hierarchies for nominal textual attributes in tabular data resp. microdata.
The work was mostly done in context of the project **[EAsyAnon](#acknowledgements)**.

## Citation

```bibtex
@InProceedings{10.1007/978-3-031-83472-1_9,
  author="Aufschl{\"a}ger, Robert
  and Wilhelm, Sebastian
  and Heigl, Michael
  and Schramm, Martin",
  editor="Chbeir, Richard
  and Ilarri, Sergio
  and Manolopoulos, Yannis
  and Revesz, Peter Z.
  and Bernardino, Jorge
  and Leung, Carson K.",
  title="ClustEm4Ano: Clustering Text Embeddings of Nominal Textual Attributes for Microdata Anonymization",
  booktitle="Database Engineered Applications",
  year="2025",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="122--137",
  abstract="This work introduces ClustEm4Ano, an anonymization pipeline that can be used for generalization and suppression-based anonymization of nominal textual tabular data. It automatically generates value generalization hierarchies (VGHs) that, in turn, can be used to generalize attributes in quasi-identifiers. The pipeline leverages embeddings to generate semantically close value generalizations through iterative clustering. We applied KMeans and Hierarchical Agglomerative Clustering on 13 different predefined text embeddings (both open and closed-source (via APIs)). Our approach is experimentally tested on a well-known benchmark dataset for anonymization: The UCI Machine Learning Repository's Adult dataset. ClustEm4Ano supports anonymization procedures by offering more possibilities compared to using arbitrarily chosen VGHs. Experiments demonstrate that these VGHs can outperform manually constructed ones in terms of downstream efficacy (especially for small k-anonymity) and therefore can foster the quality of anonymized datasets. Our implementation is made public.",
  isbn="978-3-031-83472-1"
}
```

## Getting started

Set up a Conda environment with Python 3.11 (clustem4ano)

```bash
conda create -n clustem4ano python=3.11
```

Look, where this conda environment is stored

```bash
conda info --envs
```

Activate the environment in Terminal and install requirements

```bash
conda activate clustem4ano
pip3 install -r requirements.txt
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11
```

## Folder structure

```
.
├── LICENSE
├── README.md
├── anonymetrics
│   ├── __init__.py
│   ├── anonymetrics.py
│   ├── infometrics.py
│   └── utilitymetrics.py
├── anonymized
│   └── experiment_0
│       ├── Anonymize.java
│       ├── Baseline.java
│       ├── evaluation.ipynb
│       └── metrics.csv
├── cluster
│   ├── __init__.py
│   └── cluster.py
├── create_vgh_adult.py
├── datasets
│   └── adult
├── evaluate
│   ├── __init__.py
│   └── evaluate.py
├── evaluation_adult.py
├── intuition.ipynb
├── preprocess
│   ├── __init__.py
│   ├── preprocess.py
│   └── vectorize.py
├── requirements.txt
├── test
│   ├── __init__.py
│   ├── test_anonymetrics.py
│   ├── test_infometrics.py
│   └── test_utilitymetrics.py

```

## Reproducibility:

create_vgh_adult.py: 
- Creates values generalization hierarchies (VGHs) for relevant attributes in the Adult dataset
- Might need to specify keys and tokens in \preprocess

evaluation_adult.py:
- Code to evaluate anonymized data from anonymized/experiment_0

anonymized/experiment_0:
- Contains results for an experiment on the Adult dataset: 
  - created VGHs depending on clustering and embedding used (from create_vgh_adult.py)
  - Code to anonymize Adult data using generated VGHs (Baseline.java, Anonymize.java)
  - anonymized data
  - experimental results (from evaluation_adult.py > metrics.csv)
  - visualizations of metrics.csv (evaluation.ipynb)

    
## Acknowledgements

The research project **EAsyAnon** (“Verbundprojekt: Empfehlungs- und Auditsystem zur Anonymisierung”, funding indicator: 16KISA128K) is funded by the European Union under the umbrella of the funding guideline “Forschungsnetzwerk Anonymisierung für eine sichere Datennutzung” from the German Federal Ministry of Education and Research (BMBF).
