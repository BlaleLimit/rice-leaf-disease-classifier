import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from os import path, remove
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from uuid import uuid4
from preprocess import *
from classify import *


# Argument parser for CLI inputs
parser = ArgumentParser(description='Classify Rice Leaf Diseases')
parser.add_argument('-i', '--in_file', help='Input filename')
parser.add_argument('-m', '--acnn_model', type=int, default=5, help='''
                Model [0] acnn_model_no_attention_50_one_whole_pickle.h5,
                Model [1] acnn_model_no_attention_50_v1.1.h5,
                Model [2] acnn_model_no_attention_50_v1.2.h5,
                Model [3] acnn_model_no_attention_model.h5,
                Model [4] acnn_model_no_attention_whole_pickle_lr_0000088.h5,
                Model [5] acnn_model_no_attention_whole_pickle_lr_00001.h5''')
parser.add_argument('-l', '--model_location', default='SAVED MODELS (H5 Files)', help='ACNN Models Location')

# Saves input image with uuid4 filename
def save_input(in_file):
    extension = in_file.split('.')

    if extension[len(extension)-1] in ['jpg', 'jpeg', 'png']:
        print('\nInput accepted')
        filename_uuid = uuid4()
        img = Image.open(in_file)
        img.save(path.join('INPUTS', f'{str(filename_uuid)}.{extension[len(extension)-1]}'))
        return f'{str(filename_uuid)}.{extension[len(extension)-1]}'
        
    else:
        print('Incorrect input or file format')
        return ''

# Deletes input image from INPUTS folder
def delete_input(input_image):
    if path.exists(f'INPUTS/{input_image}'):
        remove(f'INPUTS/{input_image}')

# Show available models from the SAVED MODELS (H5 Files) folder
def show_models(model_location):
    for index, model in enumerate(Path(model_location).iterdir()):
        if str(model).endswith('.h5'):
            backslash_char = "\\"
            print(f'Model [{index}] {str(model).split(backslash_char)[1]}')


def write_text(chosen_model, prediction_result):
    # remove result.txt if it exists
    if path.exists('OUTPUTS/result.txt'):
        remove('OUTPUTS/result.txt')

    # create result.txt to store results
    f = open('OUTPUTS/result.txt', 'w+')
    try:
        f.write(chosen_model)
        f.write(prediction_result)
    finally:
        f.close()

# Generate prediction
def generate_prediction(input_image, acnn_model, model_location):
    model, chosen_model = get_model(acnn_model, model_location)
    processed_image = preprocessing(input_image)
    prediction = predict_input(processed_image, model)
    prediction_result = show_prediction(prediction)
    return chosen_model, prediction_result

# Main method
def main(in_file, acnn_model, model_location):
    try:
        input_image = save_input(in_file)
        if input_image != '':
            show_models(model_location)
            chosen_model, prediction_result = generate_prediction(input_image, acnn_model, model_location)
            write_text(chosen_model, prediction_result)
            delete_input(input_image)
        
    except:
        print('Error occurred')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.in_file, args.acnn_model, args.model_location)