# Skin Cancer Risk Analyzer

A Streamlit-based tool for preliminary skin lesion screening using computer vision.

## Overview
Analyzes uploaded skin lesion images using OpenCV to quantify morphological features 
(asymmetry, border irregularity, color variance, diameter) and estimates a risk score — 
inspired by the clinical ABCD rule for melanoma screening.

## Features
- Hair removal preprocessing (morphological blackhat + inpainting)
- Lesion segmentation via Otsu thresholding
- Risk scoring from 4 weighted morphometric features
- Interactive radar chart + gauge visualization (Plotly)
- Downloadable text report

## Tech Stack
Python, Streamlit, OpenCV, NumPy, Plotly, Pillow

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer
This is an academic/educational tool and NOT a diagnostic medical device. 
It does not replace professional dermatological evaluation.

## Author
Nikita Khandare, M.Sc. Bioinformatics, DES Pune University
