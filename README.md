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
├── anonymetrics       
│   ├── anonymetrics.py
│   ├── infometrics.py
│   ├── __init__.py
│   └── utilitymetrics.py
├── anonymized
│   └── experiment_0
├── cluster_attribute.py
├── create_vgh.py
├── datasets
│   └── adult
├── embedding_pics
│   └── adult
├── evaluate
├── evaluation_adult.py
├── .gitignore
├── __init__.py
├── intuition.ipynb
├── LICENSE
├── preprocess
├── README.md
├── requirements.txt
├── test

```

## Reproducibility:

create_vgh_adult.py: Creates values generalization hierarchies (VGHs) for relevant attributes in the Adult dataset

anonymized/experiment_0: 
- Contains created VGHs depending on clustering and embedding used
- Contains Java Code to anonymize Adult data using generated VGHs
- Contains anonymized data

evaluation_adult.py:
- Code to evaluate anonymized data from anonymized/experiment_0


## Acknowledgements

The research project **EAsyAnon** (“Verbundprojekt: Empfehlungs- und Auditsystem zur Anonymisierung”, funding indicator: 16KISA128K) is funded by the European Union under the umbrella of the funding guideline “Forschungsnetzwerk Anonymisierung für eine sichere Datennutzung” from the German Federal Ministry of Education and Research (BMBF).
