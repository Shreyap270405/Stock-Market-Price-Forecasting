# utils.py
import os
from tensorflow.keras.models import load_model as keras_load_model

def save_model(model, path="saved_model/stock_model.h5"):
    """
    Saves the trained model to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"✅ Model saved at: {path}")

def load_model(path="saved_model/stock_model.h5"):
    """
    Loads the trained model from disk if it exists.
    Returns:
        model: Loaded Keras model, or None if not found.
    """
    if os.path.exists(path):
        print(f"✅ Loading model from: {path}")
        return keras_load_model(path)
    else:
        print("⚠️ No saved model found, please train first.")
        return None
