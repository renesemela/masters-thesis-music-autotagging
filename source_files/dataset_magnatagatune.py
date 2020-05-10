"""
========================= !!! READ ME !!! =========================
Use this command to get script description
    python dataset_magnatagatune.py --help
Make sure you have installed all requirements from requirements.txt
===================================================================
"""

# Libraries: Import global
import argparse

# Function: Build dataset database
def build_db():
    # Libraries: Import local
    import numpy as np
    import pandas as pd
    import sqlite3
    import re

    # Libraries: Import custom
    from masters.paths import path_dataset_mtt, path_dataset_mtt_db
    from masters.tools_dataset import folderstruct_mtt_dataset

    # Check if required folder structure is present
    print('Checking required folder structure.')
    folderstruct_mtt_dataset()

    print('Building database.')

    # Load annotations from CSV file to pd.DataFrame
    annotations = pd.read_csv(path_dataset_mtt + 'annotations_final.csv', sep='\t')

    # Rename columns to match dataset naming
    annotations = annotations.rename(columns={
        'clip_id': 'id_dataset', 
        'mp3_path': 'path_mp3' 
    })

    # Make separate metadata
    metadata = annotations[['id_dataset', 'path_mp3']] # Save metadata for future use
    annotations = annotations.drop(['id_dataset', 'path_mp3'], axis=1) # Drop metadata from 'annotations'

    # Merge synonymous tags (list)
    tags_synonyms = [
        ['beat', 'beats'],
        ['chant', 'chanting'],
        ['choir', 'choral'],
        ['classical', 'clasical', 'classic'],
        ['drum', 'drums'],
        ['electro', 'electronic', 'electronica', 'electric'],
        ['fast', 'fast beat', 'quick'],
        ['female', 'female singer', 'female singing', 'female vocals', 'female vocal', 'female voice', 'woman', 'woman singing', 'women'],
        ['flute', 'flutes'],
        ['guitar', 'guitars'],
        ['hard', 'hard rock'],
        ['harpsichord', 'harpsicord'],
        ['heavy', 'heavy metal', 'metal'],
        ['horn', 'horns'],
        ['india', 'indian'],
        ['jazz', 'jazzy'],
        ['male', 'male singer', 'male vocal', 'male vocals', 'male voice', 'man', 'man singing', 'men'],
        ['no beat', 'no drums'],
        ['no singer', 'no singing', 'no vocal','no vocals', 'no voice', 'no voices', 'instrumental'],
        ['opera', 'operatic'],
        ['orchestra', 'orchestral'],
        ['quiet', 'silence'],
        ['singer', 'singing'],
        ['space', 'spacey'],
        ['string', 'strings'],
        ['synth', 'synthesizer'],
        ['violin', 'violins'],
        ['vocal', 'vocals', 'voice', 'voices'],
        ['strange', 'weird']]
    # Merge similar tags
    for tags_synonyms_current in tags_synonyms:
        annotations[tags_synonyms_current[0]] = annotations[tags_synonyms_current].max(axis=1)
        annotations = annotations.drop(tags_synonyms_current[1:], axis=1)

    # # Make histogram of tags in dataset
    tags_hist = annotations.sum(axis=0)
    tags_hist = tags_hist.sort_values(axis=0, ascending=False) # Sort frome the most common tag to the least common

    # Pick only 50 most frequented tags
    n_tags = 50
    tags_to_remove = list(tags_hist.index[n_tags:]) # Making list of tags to be removed
    annotations = annotations.drop(tags_to_remove, axis=1) # Removing tags from 'annotations'
    
    # Remove uppercase letters and spaces in tags names
    toptags_list_cleaned = list(annotations.columns)
    toptags_list_cleaned = [x.lower() for x in toptags_list_cleaned]
    # Remove spaces and '-' and replace with '_'
    for i in range(len(toptags_list_cleaned)):
        tag_current_split = re.split(' |-', toptags_list_cleaned[i])
        if len(tag_current_split) > 1:
            temp_string = tag_current_split[0]
            for j in range(1, len(tag_current_split)):
                temp_string = temp_string + '_' + tag_current_split[j]
                toptags_list_cleaned[i] = temp_string
    # Update annotation columns with new ones
    annotations.columns = toptags_list_cleaned

    # Change dtype of annotations to integer before inserting to DB
    annotations = annotations.astype('int8')

    # Put back metadata to annotations
    annotations['id_dataset'] = metadata['id_dataset']
    
    # Prepare tags names for SQL command
    tags_sql_list = ['"' + x + '", ' for x in annotations.columns]
    tags_sql_string = ''.join(tags_sql_list)
    tags_sql_string = tags_sql_string[:len(tags_sql_string)-2]

    # Prepare metadata names for SQL command
    tags_sql_list = ['"' + x + '", ' for x in metadata.columns]
    metadata_sql_string = ''.join(tags_sql_list)
    metadata_sql_string = metadata_sql_string[:len(metadata_sql_string)-2]

    # Create database file and open connection
    conn = sqlite3.connect(path_dataset_mtt_db)
    c = conn.cursor()

    # Create separate tables for tags and metadata
    c.execute('CREATE TABLE IF NOT EXISTS tags (' + tags_sql_string + ', PRIMARY KEY("id_dataset"))')
    conn.commit()
    c.execute('CREATE TABLE IF NOT EXISTS metadata (' + metadata_sql_string + ', PRIMARY KEY("id_dataset"))')
    conn.commit()

    # Insert tags and metadata to the database and close connection
    annotations.to_sql('tags', conn, if_exists = 'replace', index = False)
    metadata.to_sql('metadata', conn, if_exists = 'replace', index= False)
    conn.close()

    print('Done.')

# Function: Convert tracks to WAV
def convert_to_wav():
    # Libraries: Import local
    import pandas as pd
    import sqlite3
    import os

    # Libraries: Import custom
    from masters.paths import path_dataset_mtt_db, path_dataset_mtt
    from masters.tools_audio import mp3_to_wav
    from masters.tools_dataset import folderstruct_mtt_dataset

    # Check if required folder structure is present
    print('Checking required folder structure.')
    folderstruct_mtt_dataset()

    # Connect to the database
    conn = sqlite3.connect(path_dataset_mtt_db)
    c = conn.cursor()

    # Select all metadata from the database and make pd.DataFrame
    c.execute('SELECT * FROM metadata')
    columns_names = [description[0] for description in c.description]
    metadata = pd.DataFrame(c.fetchall(), columns=columns_names)

    # Check if any files are present in the tracks folder
    dir_tracks = os.listdir(path_dataset_mtt + 'tracks_wav/')
    dir_tracks = [str(i.split('.', 1)[0]) for i in dir_tracks]

    # Convert each track to WAV
    i = 0
    while i < len(metadata):
        print('\nTrack ' + str(i+1) + '/' + str(len(metadata)))
        print('Converting track.')
        id_dataset = str(metadata['id_dataset'][i])
        if id_dataset not in dir_tracks:
            path_mp3 = metadata['path_mp3'][i]
            path_track_mp3 = path_dataset_mtt + 'tracks_mp3/' + path_mp3
            path_track_wav = path_dataset_mtt + 'tracks_wav/' + id_dataset + '.wav'
            mp3_to_wav(path_track_mp3, path_track_wav)
            print('Successfully converted.')
            i = i + 1
        else:
            print('Track is already converted: ' + id_dataset + '.wav')
            i = i + 1
    
    # Check if converting was successful
    dir_tracks = os.listdir(path_dataset_mtt + 'tracks_wav/')
    dir_tracks = [str(i.split('.', 1)[0]) for i in dir_tracks]
    for i in range(len(metadata['id_dataset'])):
        id_dataset = str(metadata['id_dataset'][i])
        if not id_dataset in dir_tracks:
            c.execute('DELETE FROM metadata WHERE "id_dataset" = "' + id_dataset + '"')
            conn.commit()
            c.execute('DELETE FROM tags WHERE "id_dataset" = "' + id_dataset + '"')
            conn.commit()
    
    # Close connection to the database file
    conn.close()

    print('Done.')

# Function: Compute mel spectrogram for each track and save
def compute_melgram():
    # Libraries: Import local
    import numpy as np
    import pandas as pd
    import os
    import sqlite3

    # Libraries: Import custom
    from masters.paths import path_dataset_mtt_db, path_dataset_mtt
    from masters.tools_audio import melgram
    from masters.tools_dataset import folderstruct_mtt_dataset

    # Check if required folder structure is present
    print('Checking required folder structure.')
    folderstruct_mtt_dataset()

    # Connect to the database
    conn = sqlite3.connect(path_dataset_mtt_db)
    c = conn.cursor()

    # Select all metadata from the database and make pd.DataFrame
    c.execute('SELECT * FROM metadata')
    columns_names = [description[0] for description in c.description]
    metadata = pd.DataFrame(c.fetchall(), columns=columns_names)

    # Check if any files are present in the melgram folder
    dir_features = os.listdir(path_dataset_mtt + 'features_melgram/')
    dir_features = [str(i.split('.', 1)[0]) for i in dir_features]

    # Compute mel spectrogram for each of tracks
    i = 0
    while i < len(metadata):
        print('\nTrack ' + str(i+1) + '/' + str(len(metadata)))
        print('Computing melgram.')
        id_dataset = str(metadata['id_dataset'][i])
        if id_dataset not in dir_features:
            path_track_wav = path_dataset_mtt + 'tracks_wav/' + id_dataset + '.wav'
            melgram_computed = melgram(path_track_wav)
            # Remove extra data if track is longer than expected
            if np.size(melgram_computed, axis=1) >= 1366:
                melgram_computed = np.delete(melgram_computed, range(1366,np.size(melgram_computed, axis=1)), 1)
                np.save(path_dataset_mtt + 'features_melgram/' + id_dataset + '.npy', melgram_computed)
                print('Successfully computed.')
            # Remove track from the database if track is shorter than expected
            elif np.size(melgram_computed, axis=1) < 1366:
                print('Error! The track is not 1366 samples long. Deleting from database.')
                c.execute('DELETE FROM metadata WHERE "id_dataset" = "' + id_dataset + '"')
                conn.commit()
                c.execute('DELETE FROM tags WHERE "id_dataset" = "' + id_dataset + '"')
                conn.commit()
            i = i + 1
        else:
            print('Melgram is already computed: ' + id_dataset + '.npy')
            i = i + 1
    
    # Close connection to the database file
    conn.close()
    
    print('Done.')

# Parser for command line switches and parameters
parser = argparse.ArgumentParser(description='This script builds MagnaTagATune Dataset. You have to download some files from http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset. Download annotations_final.csv file and place it to the folder ./datasets/magnatagatune_dataset. Download dataset MP3 files and place them to the folder ./datasets/magnatagatune_dataset/tracks_mp3. Some of the MP3 paths are too long for Windows (path limit), so it is better to use something like C:/temp folder for running this script.')
parser.add_argument('--build_db', action='store_true', help='Step 1: Load data from dataset CSV file, preprocess data and save as SQLite database file.')
parser.add_argument('--convert_to_wav', action='store_true', help='Step 2: Convert MP3 files to WAV.')
parser.add_argument('--compute_melgram', action='store_true', help='Step 3: Compute and save mel spectrogram for all tracks.')
args = parser.parse_args()

# Run step by step if no switches or parameters are provided
if args.build_db == False and args.convert_to_wav == False and args.compute_melgram == False:
    build_db()
    convert_to_wav()
    compute_melgram()

# Definition of what each switch or parameter does
for arg_current in vars(args):
    if arg_current == 'build_db' and args.build_db == True:
        build_db()
    elif arg_current == 'convert_to_wav' and args.convert_to_wav == True:
        convert_to_wav()
    elif arg_current == 'compute_melgram' and args.compute_melgram == True:
        compute_melgram()