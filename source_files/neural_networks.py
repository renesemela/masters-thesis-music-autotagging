"""
========================= !!! READ ME !!! =========================
Use this command to get script description
    python neural_networks.py --help
Make sure you have installed all requirements from requirements.txt
===================================================================
"""

# Random seed for reproducibility (must be first thing in the script)
import numpy as np
np.random.seed(0)

# Libraries: Import global
import argparse

# Function: Train and test neural network architecture on dataset
def train_test_network(architecture, dataset):
    # Libraries: Import local
    import sqlite3
    import pandas as pd
    from contextlib import redirect_stdout
    from skmultilearn.model_selection import iterative_train_test_split
    from keras.callbacks import ModelCheckpoint

    # Libraries: Import custom
    from masters.paths import path_dataset_lastfm2020, path_dataset_lastfm2020_db, path_dataset_mtt, path_dataset_mtt_db
    from masters.tools_system import makedir, currdate
    from masters.tools_neural import datagen, folderstruct_network, model_custom_callback, model_build_fcnn3, model_build_fcnn4, model_build_fcnn5, model_build_fcnn6, model_build_crnn3, model_build_crnn4, model_build_crnn5, model_build_crnn6

    # Variables: Set local (can be edited)
    epochs = 400
    batch_size = 32

    # Select dataset
    if dataset == 'magnatagatune':
        path_features = path_dataset_mtt + 'features_melgram/'
        path_dataset_db = path_dataset_mtt_db
    elif dataset == 'lastfm':
        path_features = path_dataset_lastfm2020 + 'features_melgram/'
        path_dataset_db = path_dataset_lastfm2020_db

    # Make work directory and make folder structure
    dir_work = folderstruct_network(architecture, dataset)

    # Connect to the database file
    conn = sqlite3.connect(path_dataset_db)
    c = conn.cursor()

    # Select all tags from the database and make pd.DataFrame
    c.execute('SELECT * FROM tags')
    columns_names = [description[0] for description in c.description]
    annotations = pd.DataFrame(c.fetchall(), columns=columns_names)
    conn.close()

    # Add '.npy' suffix to 'id_dataset' to make paths to mel spectrogram NPY files
    x = np.array([path_features + str(substr) + '.npy' for substr in list(annotations['id_dataset'])])
    x = np.reshape(x, (len(x),1))

    # Drop 'id_dataset' from annotations to get clean one-hot array
    y = np.array(annotations.drop(['id_dataset'], axis=1))

    # Get number of tags (classes) from annotations
    n_classes = np.size(y,1)

    # Split dataset 70:15:15
    # x_train, x_valtest, y_train, y_valtest = train_test_split(x, y, test_size=0.30)
    x_train, y_train, x_valtest, y_valtest = iterative_train_test_split(x, y, test_size=0.30)
    # x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=0.50)
    x_val, y_val, x_test, y_test = iterative_train_test_split(x_valtest, y_valtest, test_size=0.50)

    # Reshape back because iterative_train_test_split returns different shape than train_test_split
    x_train = np.reshape(x_train,(len(x_train),))
    x_val = np.reshape(x_val,(len(x_val),))
    x_test = np.reshape(x_test,(len(x_test),))

    # Save dataset split to NPZ file
    np.savez(dir_work + 'dataset/dataset_split.npz', x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)

    # Make generators of train, validation and test data
    gen_train = datagen(x_train, y_train, batch_size)
    gen_val = datagen(x_val, y_val, batch_size)
    gen_test = datagen(x_test, y_test, batch_size)

    # Define training callbacks
    checkpoint = ModelCheckpoint(dir_work + 'model/model_checkpoint_ep{epoch:03d}_valacc{val_acc:.4f}_valloss{val_loss:.4f}.hdf5', verbose=1, save_best_only=False)
    custom_callback = model_custom_callback(gen_val=gen_val, gen_test=gen_test, path_model=dir_work+'model/', path_test_results=dir_work+'test/', path_results=dir_work+'results/', path_dataset_db=path_dataset_db, verbose=1)
    callbacks_list = [checkpoint, custom_callback]

    # Load model architecture
    if architecture == 'fcnn3':
        model = model_build_fcnn3(n_classes)
    if architecture == 'fcnn4':
        model = model_build_fcnn4(n_classes)
    if architecture == 'fcnn5':
        model = model_build_fcnn5(n_classes)
    if architecture == 'fcnn6':
        model = model_build_fcnn6(n_classes)
    if architecture == 'crnn3':
        model = model_build_crnn3(n_classes)
    if architecture == 'crnn4':
        model = model_build_crnn4(n_classes)
    if architecture == 'crnn5':
        model = model_build_crnn5(n_classes)
    if architecture == 'crnn6':
        model = model_build_crnn6(n_classes)
    
    # Show and save model summary
    model.summary()
    with open(dir_work + 'results/model_architecture.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()

    # Train model
    model.fit_generator(generator=gen_train, validation_data=gen_val, epochs=epochs, callbacks=callbacks_list, shuffle=True)

# Parser for command line switches and parameters
parser = argparse.ArgumentParser(description='This script trains and tests specific neural network architecture on specific dataset.')
parser.add_argument('--architecture', action='store', choices=['fcnn3', 'fcnn4', 'fcnn5', 'fcnn6', 'crnn3', 'crnn4', 'crnn5', 'crnn6'], help='Select desired neural network architecture.')
parser.add_argument('--dataset', action='store', choices=['magnatagatune', 'lastfm'], help='Select desired dataset.')
args = parser.parse_args()

# Run crnn4 and magnatagatune if no switches or parameters are provided
if args.architecture == None or args.dataset == None:
    print('Choosing default configuration: --architecture crnn4 --dataset magnatagatune.')
    train_test_network('crnn4', 'magnatagatune')
else:
    train_test_network(args.architecture, args.dataset)