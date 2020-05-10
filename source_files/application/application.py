"""
========================= !!! READ ME !!! =========================
This is tkinter application for testing neural networks 
architectures. 
Make sure you have installed all requirements from requirements.txt
===================================================================
"""

# Libraries: Import global
import numpy as np
from tkinter import Tk, filedialog, Button, Label, LabelFrame, IntVar, Radiobutton, Listbox, Text, Toplevel, Menu, PhotoImage, Canvas
from tkinter.ttk import Progressbar
from keras.models import load_model
from math import floor

# Libraries: Import global custom
from masters.tools_audio import mp3_to_wav, melgram

# Variables: Set global
tag_names_mtt = ['singer', 'harpsichord', 'sitar', 'heavy', 'foreign', 'no_piano', 'classical', 'female', 'jazz', 'guitar', 'quiet', 'no_beat', 'solo', 'folk', 'ambient', 'new_age', 'synth', 'drum', 'bass', 'loud', 'string', 'opera', 'fast', 'country', 'violin', 'electro', 'trance', 'chant', 'strange', 'modern', 'hard', 'harp', 'pop', 'piano', 'orchestra', 'eastern', 'slow', 'male', 'vocal', 'no_singer', 'india', 'rock', 'dance', 'cello', 'techno', 'flute', 'beat', 'soft', 'choir', 'baroque']
tag_names_lastfm100 = ['rock', 'electronic', 'alternative', 'indie', 'pop', 'female_vocalists', 'metal', 'alternative_rock', 'jazz', 'classic_rock', 'ambient', 'experimental', 'folk', 'punk', 'indie_rock', 'hard_rock', 'instrumental', 'black_metal', 'singer_songwriter', 'dance', '80s', 'progressive_rock', 'death_metal', 'heavy_metal', 'hardcore', 'british', 'soul', 'chillout', 'classical', 'rap', 'industrial', 'soundtrack', 'punk_rock', 'blues', 'thrash_metal', '90s', 'acoustic', 'metalcore', 'psychedelic', 'japanese', 'post_rock', 'progressive_metal', 'german', 'hip_hop', 'funk', 'new_wave', 'trance', 'house', 'piano', 'reggae', 'american', 'trip_hop', 'techno', 'post_punk', '70s', 'electro', 'indie_pop', '60s', 'rnb', 'country', 'melodic_death_metal', 'power_metal', 'downtempo', 'male_vocalists', 'emo', 'post_hardcore', 'doom_metal', 'oldies', 'love', 'beautiful', 'psychedelic_rock', '00s', 'french', 'synthpop', 'gothic_metal', 'russian', 'grunge', 'gothic', 'guitar', 'idm', 'britpop', 'dark_ambient', 'noise', 'cover', 'swedish', 'mellow', 'screamo', 'lounge', 'symphonic_metal', 'grindcore', 'nu_metal', 'j_pop', 'pop_rock', 'polish', 'chill', 'blues_rock', 'avant_garde', 'drum_and_bass', 'new_age', 'ska']
tag_names_lastfm50 = ['rock', 'electronic', 'indie', 'alternative', 'pop', 'female_vocalists', 'alternative_rock', 'indie_rock', 'metal', 'experimental', 'ambient', 'instrumental', 'chillout', 'singer_songwriter', 'punk', 'folk', 'classic_rock', 'hip_hop', 'dance', 'hard_rock', 'british', 'death_metal', 'industrial', '80s', 'punk_rock', 'acoustic', 'indie_pop', 'heavy_metal', 'metalcore', 'hardcore', '90s', 'soul', 'progressive_rock', 'emo', 'beautiful', 'jazz', 'mellow', 'piano', 'psychedelic', 'downtempo', 'post_punk', 'progressive_metal', 'synthpop', 'electro', 'soundtrack', 'chill', 'love', 'post_hardcore', 'gothic', 'trip_hop']
models_path = './data/models/' 
models_list_mtt = ['fcnn3_magnatagatune.hdf5', 'fcnn4_magnatagatune.hdf5', 'fcnn5_magnatagatune.hdf5', 'fcnn6_magnatagatune.hdf5', 'crnn3_magnatagatune.hdf5', 'crnn4_magnatagatune.hdf5', 'crnn5_magnatagatune.hdf5', 'crnn6_magnatagatune.hdf5']
models_list_lastfm50 = ['', '', 'fcnn5_lastfm50.hdf5', '', '', 'crnn4_lastfm50.hdf5', 'crnn5_lastfm50.hdf5', '']
models_list_lastfm100 = ['', '', 'fcnn5_lastfm100.hdf5', '', '', 'crnn4_lastfm100.hdf5', 'crnn5_lastfm100.hdf5', '']

# Function: Update status in status window
def status_update(text):
    txt_status.insert('end', text + '\n')

# Function: Load audio file from file open dialog
def load_audio_file():
    # Reset status window
    txt_status.delete('1.0','end')
    # Reset progress bar
    pb_predict.config(value = 0)
    # Set global variable for audio path, load audio path and update status
    global path_audio
    path_audio = filedialog.askopenfilename(title='Select Audio File', filetypes = (('MP3 Files','*.mp3'),('WAV Files','*.wav'),('All Files (not tested)','*.*')))
    if path_audio == '':
        status_update('No audio was loaded. Please select audio file!')
    else:    
        status_update('Loaded audio file: ' + path_audio)

# Function: Main function for prediction
def predict_tags():
    if 'path_audio' not in globals() or path_audio == '':
        status_update('No audio was loaded. Please select audio file!')
    else:
        # Load variables based on dataset selection
        if dataset_choice.get() == 0:
            models_list = models_list_mtt
            tag_names = tag_names_mtt
        elif dataset_choice.get() == 1 and dataset_ntags.get() == 50:
            models_list = models_list_lastfm50
            tag_names = tag_names_lastfm50
        elif dataset_choice.get() == 1 and dataset_ntags.get() == 100:
            models_list = models_list_lastfm100
            tag_names = tag_names_lastfm100
        # Reset listbox with results, reset progress bar, update status and update whole window
        lb_results.delete(0, 'end')
        pb_predict.config(value = 0)
        status_update('\nConverting track.')
        window.update()
        # Convert audio file to WAV and save as temporary WAV file
        path_wav = './data/current_song.wav'
        mp3_to_wav(path_audio, path_wav)
        # Compute mel spectrogram
        status_update('Computing mel spectrogram.')
        window.update()
        melgram_computed = melgram(path_wav)
        melgram_computed = np.reshape(melgram_computed, (1,96,np.size(melgram_computed,axis=1)))
        # Divide mel spectrogram into segments with length 1366 which can be processed by neural network
        n_segments = floor(np.size(melgram_computed,axis=2) / 1366)
        predictions_list = np.zeros((dataset_ntags.get(), n_segments))
        # Predict tags for each segment and save to 'predictions_list'
        status_update('Predicting tags for ' + str(n_segments) + '*30 seconds.')
        for i in range(n_segments):
            status_update('Predicting segment: ' + str(i+1) + '/' + str(n_segments))
            pb_predict.config(value = (90/n_segments)*(i+1))
            window.update()
            melgram_current = melgram_computed[:,:,i*(1366):(i+1)*1366]
            model = load_model(models_path + models_list[model_choice.get()])
            predictions = model.predict(melgram_current, batch_size=1, verbose=0)
            predictions = np.round((predictions * 100))[0]
            predictions = np.transpose(predictions)
            predictions_list[:,i] = predictions 
        # Save tag to results if tag probability is more than 50 %
        results_list = []
        for i in range(np.size(predictions_list,axis=0)):
            percentage = np.max(predictions_list[i,:])
            if percentage >= 10:
                results_list.append([tag_names[i], int(percentage)])
        results_list = sorted(results_list, key=lambda results_list: results_list[1], reverse=True)
        for i in range(len(results_list)):
            lb_results.insert(i, results_list[i][0] + ': ' + str(results_list[i][1]) + ' %')
        pb_predict.config(value = 100)
        status_update('Done.')

# Function: Enable or disable GUI options for Last.fm Dataset
def lastfm_limited():
    rbt_fcnn3['state'] = 'disabled'
    rbt_fcnn4['state'] = 'disabled'
    # rbt_fcnn5['state'] = 'disabled'
    rbt_fcnn6['state'] = 'disabled'
    rbt_crnn3['state'] = 'disabled'
    # rbt_crnn4['state'] = 'disabled'
    # rbt_crnn5['state'] = 'disabled'
    rbt_crnn6['state'] = 'disabled'
    model_choice.set(6)
    rbtn_50tags['state'] = 'normal'
    rbtn_100tags['state'] = 'normal'
    dataset_ntags.set(100)

# Function: Enable or disable GUI options for MagnaTagATune Dataset
def mtt_unlimited():
    rbt_fcnn3['state'] = 'normal'
    rbt_fcnn4['state'] = 'normal'
    rbt_fcnn5['state'] = 'normal'
    rbt_fcnn6['state'] = 'normal'
    rbt_crnn3['state'] = 'normal'
    rbt_crnn5['state'] = 'normal'
    rbt_crnn6['state'] = 'normal'
    model_choice.set(5)
    rbtn_50tags['state'] = 'disabled'
    rbtn_100tags['state'] = 'disabled'
    dataset_ntags.set(50)

# Function: Build Help window
def window_help():
    popup = Toplevel()
    frame_help = LabelFrame(popup, text='Help', font=['TkDefaultFont', 9, 'bold'])
    frame_help.grid(column=1, row=1, rowspan=2, sticky='N')
    txt_help = Label(frame_help, height=17, width=40, text='1. Load any type of audio file (MP3, WAV, etc.). There is ffmpeg used for converting, so feel free to use any kind of audio format supported by ffmpeg.\n\n2. Choose preferred dataset and neural network architecture and hit "Predict tags".\n\n3. Predicting may take a while depending on your hardware. NVIDIA GPU with CUDA support for TensorFlow is recommended.', wraplengt=200)
    txt_help.grid()
    popup.config(borderwidth=10)
    popup.grab_set()

# Function: Build About window
def window_about():
    popup = Toplevel()
    frame_about = LabelFrame(popup, text='About', font=['TkDefaultFont', 9, 'bold'])
    frame_about.grid(column=1, row=1, rowspan=2, sticky='N')
    Label(frame_about, text='This app is one of the results of my master\'s thesis.', wraplengt=200).grid(column=0, row=0, columnspan=2)
    Label(frame_about, text='\nUniversity: \nFaculty: \nField of Study: \nThesis: \nSupervisor: \n Author: ', font=['TkDefaultFont', 9, 'bold'], justify='right').grid(column=0, row=1)
    Label(frame_about, text='\nBrno University of Technology \nFaculty of Electrical Engineering and Communication \nAudio Engineering \nAutomatic Tagging of Musical Compositions Using Machine Learning Methods \nIng. Tomáš Kiska \nBc. René Semela', justify='left').grid(column=1, row=1)
    popup.config(borderwidth=10)
    popup.grab_set()

# Build main windows 
window = Tk()
window.title('MUSIC AUTO-TAGGING TOOL')
Label(window, text='MUSIC AUTO-TAGGING TOOL', font=['TkDefaultFont', 16, 'bold'], justify='left').grid(column=0, row=0, columnspan=3, rowspan=2)

# Horizontal upper menu
menu_upper = Menu(window)
menu_file = Menu(menu_upper, tearoff=0)
menu_upper.add_cascade(label='File', menu=menu_file)
menu_file.add_command(label='Load audio file', command=load_audio_file)
menu_file.add_command(label='Predict tags', command=predict_tags)
menu_file.add_separator()
menu_file.add_command(label='Exit', command=window.destroy)
menu_help = Menu(menu_upper, tearoff=0)
menu_upper.add_cascade(label='Help',menu=menu_help)
menu_help.add_command(label='Help', command=window_help)
menu_help.add_command(label='About', command=window_about)
window.config(menu=menu_upper)

# Dataset choice
frame_dataset_choice = LabelFrame(window, text='Choose dataset', font=['TkDefaultFont', 9, 'bold'], labelanchor='n', width=25, padx=5, pady=5)
frame_dataset_choice.grid(column=0, row=2, sticky='n')
dataset_choice = IntVar()
dataset_choice.set(0)
Radiobutton(frame_dataset_choice, text='MagnaTagATune Dataset', variable=dataset_choice, value = 0, command=mtt_unlimited, width=22).grid(column=0, row=0)
Radiobutton(frame_dataset_choice, text='Last.fm Dataset 2020', variable=dataset_choice, value = 1, command=lastfm_limited, width=22).grid(column=0, row=1)
dataset_ntags = IntVar()
dataset_ntags.set(50)
frame_dataset_lastfm = LabelFrame(frame_dataset_choice, text='Choose number of tags', font=['TkDefaultFont', 9, 'bold'], labelanchor='n', width=16)
frame_dataset_lastfm.grid(column=0, row=2, sticky='n')
rbtn_50tags = Radiobutton(frame_dataset_lastfm, text='50', variable=dataset_ntags, value=50, width=5, state='disabled')
rbtn_50tags.grid(column=0, row=0)
rbtn_100tags = Radiobutton(frame_dataset_lastfm, text='100', variable=dataset_ntags, value=100, width=5, state='disabled')
rbtn_100tags.grid(column=1, row=0)

# Neural network choice
frame_model_choice = LabelFrame(window, text='Choose type of neural network', font=['TkDefaultFont', 9, 'bold'], labelanchor='n', width=25, padx=5, pady=5)
frame_model_choice.grid(column=0, row=3, sticky='s')
model_choice = IntVar()
model_choice.set(5)
rbt_fcnn3 = Radiobutton(frame_model_choice, text='FCNN3', variable=model_choice, value = 0, width=22)
rbt_fcnn3.grid(column=0, row=0)
rbt_fcnn4 = Radiobutton(frame_model_choice, text='FCNN4', variable=model_choice, value = 1, width=22)
rbt_fcnn4.grid(column=0, row=1)
rbt_fcnn5 = Radiobutton(frame_model_choice, text='FCNN5', variable=model_choice, value = 2, width=22)
rbt_fcnn5.grid(column=0, row=2)
rbt_fcnn6 = Radiobutton(frame_model_choice, text='FCNN6', variable=model_choice, value = 3, width=22)
rbt_fcnn6.grid(column=0, row=3)
rbt_crnn3 = Radiobutton(frame_model_choice, text='CRNN3', variable=model_choice, value = 4, width=22)
rbt_crnn3.grid(column=0, row=4)
rbt_crnn4 = Radiobutton(frame_model_choice, text='CRNN4', variable=model_choice, value = 5, width=22)
rbt_crnn4.grid(column=0, row=5)
rbt_crnn5 = Radiobutton(frame_model_choice, text='CRNN5', variable=model_choice, value = 6, width=22)
rbt_crnn5.grid(column=0, row=6)
rbt_crnn6 = Radiobutton(frame_model_choice, text='CRNN6', variable=model_choice, value = 7, width=22)
rbt_crnn6.grid(column=0, row=7)

# Status text area
frame_status = LabelFrame(window, text='Status', font=['TkDefaultFont', 9, 'bold'], labelanchor='n', padx=5, pady=5)
frame_status.grid(column=1, row=2, rowspan=2, sticky='n')
txt_status = Text(frame_status, height=21, width=40)
txt_status.grid()

# Results listbox
frame_results = LabelFrame(window, text='Tags probability', font=['TkDefaultFont', 9, 'bold'], labelanchor='n', padx=5, pady=5)
frame_results.grid(column=2, row=2, rowspan=2, sticky='n')
lb_results = Listbox(frame_results, height=21, width=30)
lb_results.grid()

# Buttons
Button(window, text='Load audio file', command=load_audio_file, width=22, font=['TkDefaultFont', 9, 'bold']).grid(column=0, row=4, padx=5, pady=5)
Button(window, text='Predict tags', command=predict_tags, width=22, font=['TkDefaultFont', 9, 'bold']).grid(column=0, row=5, padx=5, pady=0)

# Progress bar
pb_predict = Progressbar(window, orient='horizontal', length=325, mode='determinate')
pb_predict.grid(column=1, row=5)

# Version label
Label(window, text='v1.0', font=['TkDefaultFont', 8, 'normal'], fg='Grey', justify='right').grid(column=2, row=5, sticky='se')

# Initial status
status_update('Predicting may take a while depending on your hardware.\nDon\'t interrupt if hangs. Be patient!')

# Build main window
window.config(borderwidth=10)
window.mainloop()