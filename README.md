# E-Commerce Product Classifier (97% F1)

This project implements a complete Data Science pipeline—from raw data acquisition via web scraping to a high-accuracy, production-ready Product Classification Model.

The goal was to build a system that automatically and accurately assigns main categories to thousands of e-commerce products using unstructured text data (name, description) and price, demonstrating robust Feature Engineering and Classification techniques on real-world, noisy data . 

## data source
The raw product data utilized in this analysis was obtained via Web Scraping using a custom solution. The code and methodology for data acquisition are detailed in its dedicated repository:

Data Source Repository: [Your Web Scraper Repo Link]


## 🌟 Key Results & Performance
The final model, a highly efficient Linear Support Vector Classifier (LinearSVC), achieved outstanding generalization and prediction capabilities on real-world e-commerce data.

Overall Accuracy: 97.0%
Weighted F1-Score: 0.97	

## Key Features
- Data cleaning and preprocessing with Pandas
- Exploratory Data Analysis (EDA) using Jupyter notebooks
- TF-IDF vectorization for text features
- Multi-class classification pipeline
- Logging for reproducibility
- Model training and prediction scripts

## Skills Demonstrated
- Python & Object-Oriented Programming
- Data preprocessing & feature engineering
- Machine Learning & model pipelines
- Logging and modular code structure
- Jupyter notebook analysis
- Version control & project organization

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

## ⚙️ How to Run the Project