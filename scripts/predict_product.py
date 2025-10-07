import os
from models.product_category_classifier import ProductCategoryClassifier
from utils.logger import setup_logger
from scripts.train_model import train_model

logger = setup_logger()

def main():
    model_dir = r'outputs\models'
    pipeline_path = os.path.join(model_dir, 'product_category_pipeline')

    # Train model if it doesn't exist
    if not os.path.exists(pipeline_path + '_classifier.pkl'):
        logger.info('Model not found. Training a new model...')
        train_model()

    # Load the pipeline
    pipeline = ProductCategoryClassifier()
    pipeline.load_pipeline(pipeline_path)

    # Predict a new product
    new_product = {
        'price': 90,
        'description': 'Gaming headset with RGB lights and high-quality sound.'
    }

    category = pipeline.predict(new_product['price'], new_product['description'])
    logger.info(f'Predicted Category: {category}')

if __name__ == '__main__':
    main()