# ClustEm4Ano
Implementation of methods for data anonymization through generalization and suppression. The implementations restrict on tabular data resp. microdata. The work was mostly done in context of the project **[EAsyAnon](#acknowledgements)**.


## Getting started

Set up a Conda environment with Python 3.11 (ClustEm4Ano)

```bash
conda create -n clustem4ano python=3.11
```

Look, where this conda environment is stored

```bash
conda info --envs
```

Activate the environment in Terminal

```bash
conda activate clustem4ano
```

Install requirements 

```bash
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
