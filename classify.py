from pathlib import Path
from numpy import argmax
from keras.models import load_model

# Gets the chosen model from SAVED MODELS (H5 Files) based on input [0-5]
def get_model(acnn_model, model_location):
    for index, model in enumerate(Path(model_location).iterdir()):
        if str(model).endswith('.h5') and index == acnn_model:
            acnn = load_model(model)
            backslash_char = "\\"
            chosen_model = f'\nThe chosen model is [{index}] {str(model).split(backslash_char)[1]}'
            print(chosen_model)
            return acnn, chosen_model
    return

# Predicts the classification of the input and returns the prediction
def predict_input(processed_image, model):
    y_pred = model.predict(processed_image)
    prediction = [argmax(element) for element in y_pred]
    return prediction

# Maps the prediction and returns the prediction as string
def show_prediction(prediction):
    classes = {0: 'Bacterial leaf blight', 1: 'Brown spot', 2: 'Leaf smut'}
    predictions = classes.get(prediction[0])
    result = f'\nThe prediction is {predictions}.'
    print(result)
    return result

