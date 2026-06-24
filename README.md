# SDSS Star / Galaxy / Quasar Classifier

ML model that classifies Stars, Galaxies and Quasars using real data from the Sloan Digital Sky Survey (100,000 real astronomical observations).

## Results
- Random Forest: 97%+ accuracy
- Gradient Boosting: 97%+ accuracy

## How it works
Telescopes measure brightness through 5 color filters (u, g, r, i, z). Combined with redshift values and color index features, the model classifies each object as a Star, Galaxy or Quasar. You can also run it and enter values manually to predict what a space object is.

## Dataset
Download star_classification.csv from Kaggle and put it in the same folder as the script:
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

## Install
pip install pandas numpy scikit-learn matplotlib seaborn

## Run
python sdss_classifier1.py

## Built with
Python, scikit-learn, pandas, matplotlib

Data source: Sloan Digital Sky Survey DR17 (public domain)
