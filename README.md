# Wine Quality Prediction App

This repository contains a Streamlit web application for predicting red wine quality based on physicochemical features.

## Project Overview

This project demonstrates a complete machine learning pipeline, from data exploration and model training to interactive web application development and cloud deployment. The goal is to predict the 'quality' score of red wines using various chemical properties.

## Dataset

The model is trained on the **Red Wine Quality Dataset** from Kaggle.
[Link to Kaggle Dataset: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009]

## Features

* **Data Exploration**: View dataset overview, sample data, and interactive filtering.
* **Visualizations**: Explore data distributions and relationships with various charts (histograms, box plots, correlation heatmap, scatter plots).
* **Model Prediction**: Get real-time wine quality predictions by entering physicochemical properties.
* **Model Performance**: Review key evaluation metrics and understand the chosen model's performance.

## Installation and Local Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/wine-quality-predictor.git](https://github.com/YOUR_GITHUB_USERNAME/wine-quality-predictor.git)
    cd wine-quality-predictor
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    conda create -n wine_env python=3.9 # Or a recent Python version
    conda activate wine_env
    ```
    (If not using Anaconda, `python -m venv venv` and `source venv/bin/activate` on Linux/macOS, or `.\venv\Scripts\activate` on Windows.)
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The app will open in your default web browser.

## Project Structure