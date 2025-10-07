# E-Commerce Product Classifier (97% F1)

This project implements a complete Data Science pipeline—from raw data acquisition via web scraping to a high-accuracy, production-ready Product Classification Model.

The goal was to build a system that automatically and accurately assigns main categories to thousands of e-commerce products using unstructured text data (name, description) and price, demonstrating robust Feature Engineering and Classification techniques on real-world, noisy data . 

---

## data source
The raw product data utilized in this analysis was obtained via Web Scraping using a custom solution. The code and methodology for data acquisition are detailed in its dedicated repository:

Data Source Repository: [https://github.com/Israaelhouch/tunisianet_scraper.git]

---

## Key Results & Performance
The final model, a highly efficient Linear Support Vector Classifier (LinearSVC), achieved outstanding generalization and prediction capabilities on real-world e-commerce data.

Overall Accuracy: 97.0%
Weighted F1-Score: 0.97	

---

## Key Features

- Comprehensive data cleaning and preprocessing using Pandas
- Exploratory Data Analysis (EDA) via Jupyter notebooks
- TF-IDF vectorization for text-based features
- Multi-class classification pipeline
- Logging and modular code structure for reproducibility
- Scripts for model training and prediction

---

## Tech stack
- Programming & Scripting: Python 3.x, Object-Oriented Programming
- Data Manipulation & Analysis: Pandas, NumPy
- Text Processing & Feature Engineering: TF-IDF Vectorizer, Scikit-learn preprocessing modules
- Machine Learning & Modeling: Scikit-learn (LinearSVC), Model Pipelines, - Multi-class Classification
- Exploratory Data Analysis (EDA): Jupyter Notebooks, Matplotlib, Seaborn
- Project Structure & Reproducibility: Modular code design, Logging
- Version Control & Collaboration: Git, GitHub

---

## Project Structure: product_category_classifier

```
├── 📁 __pycache__/ 🚫 (auto-hidden)
├── 📁 data/
│   ├── 📁 cleaned/
│   │   └── 📄 preprocessed_data.csv
│   └── 📁 raw/
│       └── 📄 products.csv
├── 📁 models/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   └── 🐍 product_category_classifier.py
├── 📁 notebooks/
│   ├── 📓 EDA.ipynb
│   └── 📓 data_cleaning.ipynb
├── 📁 outputs/
│   └── 📁 logs/
│       └── 📋 training.log 🚫 (auto-hidden)
├── 📁 scripts/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   ├── 🐍 predict_product.py
│   └── 🐍 train_model.py
├── 📁 utils/
│   ├── 📁 __pycache__/ 🚫 (auto-hidden)
│   └── 🐍 logger.py
├── 🚫 .gitignore
├── 📖 README.md
└── 📄 requirements.txt
```
---

## How to Run

Set up and run the **Product Category Classifier** in a few simple steps.

1. Clone the Repository:

```bash
git clone https://github.com/Israaelhouch/product_category_classifier.git
cd product-category-classifier
```

2. Set Up a Virtual Environment (Optional but Recommended):
```bash
python -m venv venv
# Activate the environment
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```
3. Install Dependencies:
```bash
pip install -r requirements.txt
```

4. Reproduce the Model: To skip the notebooks and train the final model:
```bash
python scripts\train_model.py
```

5. Classify new products. The script will automatically train the model if it doesn’t exist:
```bash
python scripts\predict_product.py
```
