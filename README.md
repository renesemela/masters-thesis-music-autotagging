# Master's Thesis: Automatic Tagging of Musical Compositions Using Machine Learning Methods 
Using artificial neural networks for music auto-tagging purposes. This whole work is written in *Python* (*Keras*, *TensorFlow*). There are 2 datasets used as a source for neural networks (*MagnaTagATune Dataset*, *Last.fm Dataset 2020*)

This repository is one of the results of my master's thesis. 

* **University:** Brno University of Technology, Czech Republic
* **Faculty:** Faculty of Electrical Engineering and Communication, Department of Telecomunications
* **Field of Study:** Audio Engineering
* **Supervisor:** Ing. Tomáš Kiska
* **Author:** Bc. René Semela


## Getting Started
These instructions will help you to get familiar with overall concept of source codes in this work.

There are two folders:
* **installation_files** - Contains prerequisites (*Python 3.7.7* and *Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019*).
* **source_files** - Contains all source codes. This folder is considered as main working folder for running scripts.

### Prerequisites
* *Microsoft Windows 10 (64-bit)*
* *Python 3.7.x (64-bit)*
* *Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 (64-bit)*
* It is recommended to use a virtual environment in ***source_files*** folder. This can be done with *virtualenv*
 Python library (with Windows command line):
```
cd source_files
pip install virtualenv
virtualenv .venv
".venv\Scripts\activate.bat"
```
* *Python libraries* from ***requirements.txt*** can be installed via *pip*:
```
pip install -r requirements.txt
```

## Running the scripts
### dataset_magnatagatune.py
This script uses [MagnaTagATune Dataset](http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset). You will need to download MP3 files from the official site ([part1](http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001), [part2](http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002), [part3](http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003)) and unzip this archive into the folder ***datasets\magnatagatune_dataset\tracks_mp3***. After this, you can run this script without any switch:
```
python dataset_magnatagatune.py
```
or you can obtain more info about switches:
```
python dataset_magnatagatune.py --help
```


### dataset_lastfm.py
This script builds a completely new Last.fm Dataset 2020 inspired by original [Last.fm Dataset](http://millionsongdataset.com/lastfm/) which is now (2020) very old and unsupported.

The script is set to build dataset from scratch using Last.fm and Spotify API for data acquisition. This step is not necessary because there is dataset database already present in ***datasets\lastfm_dataset_2020*** and all you need to do is run this script with some switches to get tracks from Spotify servers (This dataset contains 100 tags and 122 877 tracks.):
```
python dataset_lastfm.py --download_spotify_preview
python dataset_lastfm.py --convert_to_wav
python dataset_lastfm.py --compute_melgram
```
More info about this script can be obtained with --help switch:
```
python dataset_lastfm.py --help
```
If you would like to build dataset from scratch, you will need to insert Last.fm and Spotify API keys into variables ***api_key_spotify*** and ***api_key_lastfm*** in the source code of this script.

### neural_networks.py
This script uses *Keras* and *TensorFlow* for training, testing and evaluating neural networks architectures. There are 8 neural networks architectures (crnn3, crnn4, crnn5, crnn6, fcnn3, fcnn4, fcnn5, fcnn6) and you can run this script as follows (parameters can be changed):
```
python neural_networks.py --architecture crnn4 --dataset magnatagatune
```
More info can be obtained via this command:
```
python neural_networks.py --help
```

### appliacation.py
You can find this script in the ***application*** folder. It is completely standalone application with GUI for testing purposes. You can use trained architectures (by me) for predicting tags on your own audio file (MP3 or WAV). You can run this app with simple command: 
```
cd application
python application.py
```
![application.png](https://github.com/renesemela/masters-thesis-music-autotagging/blob/master/images/application.png?raw=true)

## Results of this work
Here are some figures of the results on both datasets.
![roc_auc_all_magnatagatune.png](https://github.com/renesemela/masters-thesis-music-autotagging/blob/master/images/roc_auc_all_magnatagatune.png?raw=true)

![roc_auc_all_lastfm.png](https://github.com/renesemela/masters-thesis-music-autotagging/blob/master/images/roc_auc_all_lastfm.png?raw=true)

## Built With
* Python
* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Librosa](https://librosa.github.io/)
* [SciPy](https://scipy.org/)
* and many more

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
* [BDALab](https://bdalab.utko.feec.vutbr.cz/) - Computational resources
* [MetaCentrum VO](https://metavo.metacentrum.cz/) - Computational resources
