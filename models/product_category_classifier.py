import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack, csr_matrix

from utils.logger import setup_logger
logger = setup_logger()


class ProductCategoryClassifier:
    def __init__(self, tfidf_max_features=10000, ngram_range=(1, 2), classifier_params=None):
        self.tfidf_max_features = tfidf_max_features
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            stop_words='english', ngram_range=ngram_range, max_features=tfidf_max_features
        )
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.classifier_params = classifier_params or {
            'random_state': 42,
            'C': 0.5,
            'class_weight': 'balanced',
            'max_iter': 2000,
        }
        self.classifier = LinearSVC(**self.classifier_params)

    # --- Data Preprocessing ---
    def preprocess(
        self,
        df,
        text_columns=['name', 'description'],
        target_column='main_category',
        numeric_columns=['price'],
    ):
        try:
            if df.isnull().any().any():
                logger.warning(
                    "Missing values detected. Filling with empty strings for text and zeros for numeric columns."
                )
                df[text_columns] = df[text_columns].fillna('')
                df[numeric_columns] = df[numeric_columns].fillna(0)

            df['combined_text'] = df[text_columns].apply(lambda x: ' '.join(x), axis=1)
            self.X_text = self.tfidf.fit_transform(df['combined_text'])

            df['price_scaled'] = self.scaler.fit_transform(df[numeric_columns].values)
            self.X_numeric = csr_matrix(df['price_scaled'].values.reshape(-1, 1))

            self.Y = self.label_encoder.fit_transform(df[target_column])
            self.feature_matrix = hstack([self.X_text, self.X_numeric])
            self.df = df
            self.target_names = self.label_encoder.classes_

            logger.info("Preprocessing completed successfully.")
            return self.feature_matrix, self.Y

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    # --- Train-Test Split ---
    def split_data(self, test_size=0.2, random_state=42):
        try:
            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
                self.feature_matrix,
                self.Y,
                test_size=test_size,
                random_state=random_state,
                stratify=self.Y,
            )
            logger.info(
                f"Data split: {self.X_train.shape[0]} train samples, {self.X_test.shape[0]} test samples."
            )
            return self.X_train, self.X_test, self.Y_train, self.Y_test

        except Exception as e:
            logger.error(f"Error during data splitting: {e}")
            raise

    # --- Model Training ---
    def train(self):
        try:
            self.classifier.fit(self.X_train, self.Y_train)
            logger.info("Model training completed successfully.")
            return self.classifier
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    # --- Evaluation ---
    def evaluate(self, plot_cm=False, save_dir="outputs/plots"):
        try:
            predictions = self.classifier.predict(self.X_test)
            report = metrics.classification_report(
                self.Y_test, predictions, target_names=self.target_names
            )
            logger.info("Classification Report:\n%s", report)

            if plot_cm:
                os.makedirs(save_dir, exist_ok=True)
                cm = confusion_matrix(self.Y_test, predictions)
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt='d',
                    cmap="Blues",
                    xticklabels=self.target_names,
                    yticklabels=self.target_names,
                    cbar=False,
                )
                plt.title("Confusion Matrix - Main Categories")
                plt.ylabel('Actual Category')
                plt.xlabel('Predicted Category')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                save_path = os.path.join(save_dir, "confusion_matrix.png")
                plt.savefig(save_path)
                plt.close()
                logger.info(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise

    # --- Inference ---
    def predict(self, price, description):
        try:
            X_text_new = self.tfidf.transform([description])
            X_price_new = csr_matrix(self.scaler.transform([[price]]))
            X_new = hstack([X_text_new, X_price_new])
            prediction_id = self.classifier.predict(X_new)[0]
            return self.label_encoder.inverse_transform([prediction_id])[0]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    # --- Save and Load ---
    def save_pipeline(self, path_prefix='outputs/models/product_category_pipeline'):
        try:
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            joblib.dump(self.tfidf, f'{path_prefix}_tfidf.pkl')
            joblib.dump(self.scaler, f'{path_prefix}_scaler.pkl')
            joblib.dump(self.label_encoder, f'{path_prefix}_label_encoder.pkl')
            joblib.dump(self.classifier, f'{path_prefix}_classifier.pkl')
            logger.info("Pipeline saved successfully.")
        except Exception as e:
            logger.error(f"Error saving pipeline: {e}")
            raise

    def load_pipeline(self, path_prefix='outputs/models/product_category_pipeline'):
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_prefix = os.path.join(base_dir, path_prefix.replace('/', os.sep))
            
            print(f"Loading pipeline from: {abs_prefix}_tfidf.pkl")

            self.tfidf = joblib.load(f'{abs_prefix}_tfidf.pkl')
            self.scaler = joblib.load(f'{abs_prefix}_scaler.pkl')
            self.label_encoder = joblib.load(f'{abs_prefix}_label_encoder.pkl')
            self.classifier = joblib.load(f'{abs_prefix}_classifier.pkl')
            
            logger.info("Pipeline loaded successfully.")
        
        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            raise
