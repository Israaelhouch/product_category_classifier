import pandas as pd
from utils.logger import setup_logger
from models.product_category_classifier import ProductCategoryClassifier

logger = setup_logger()

def train_model():
    try:
        logger.info('Starting training process...')
        df = pd.read_csv('data/cleaned/preprocessed_data.csv')
        pipeline = ProductCategoryClassifier()

        # Preprocess data
        pipeline.preprocess(df)

        # Train-Test split
        pipeline.split_data()

        # Train model
        pipeline.train()

        # Evaluate model immediately after training
        pipeline.evaluate(plot_cm=False)  # Set plot_cm=True if you want the CM saved

        # Save trained pipeline
        pipeline.save_pipeline('outputs/models/product_category_pipeline')
        logger.info('Training complete. Model saved successfully.')

    except Exception as e:
        logger.error(f"Error in training process: {e}")

if __name__ == '__main__':
    train_model()