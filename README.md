# RiceLeafDiseaseClassifier

This folder contains the dataset, past model versions, logs performance records, preprocessing methods, and saved models (H5 Files).

Folders
* DATASET - Contains the pickle files that were used in training, validating, and testing of the built model.
* PAST VERSIONS OF RECOGNITION SYTEMS - Contains the past version of the Rice Leaf Recognition System.
* PREPROCESSING METHODS - Contains the Jupyter notebooks that were used to preprocess and generate the datasets.
* PERFORMANCE RECORDS OF MODELS - Showcases the performance of various versions of the model during training time.
* SAVED MODEL (H5 Files) - Contains the saved trained models that can be loaded to a Jupyter notebook.
* SYSTEM - Contains the Rice_Leaf_Recognition_System.ipynb and Rice_Leaf_Recognition_Comparison.ipynb

SYSTEM folder
* Rice_Leaf_Recognition_System.ipynb - Jupyter notebook consisting the code that will run two models namely, (a) ACNN Model WITH Visual Attention and (b) ACNN Model WITHOUT Visual Attention.
* Rice_Leaf_Recognition_Comparison.ipynb - Jupyter notebook comparing the performance of the generated *.h5 models.
* RLRS.py - Python file that accepts image inputs (jpg, jpeg, png) through CLI. The models can be selected based on the available models in SAVED MODELS (H5 files). The system returns the predicted output.

Requirements
* You can install the packages in the requirements.txt file.

