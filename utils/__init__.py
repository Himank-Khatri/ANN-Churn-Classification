import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_artifacts():    
    with open('artifacts/gender_label_encoder.pkl', 'rb') as file:
        gender_label_encoder = pickle.load(file)
    
    with open('artifacts/geo_oh_encoder.pkl', 'rb') as file:
        geo_oh_encoder = pickle.load(file)
    
    with open('artifacts/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    model = load_model('artifacts/model.h5')

    return gender_label_encoder, geo_oh_encoder, scaler, model