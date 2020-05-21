@echo off
echo Installing virtualenv library for Python...
pip install virtualenv

echo. & echo Creating virtualenv in the folder .venv...
virtualenv .venv

echo. & echo Activating virtualenv... & ".venv\Scripts\activate.bat" & echo. & echo Instaling required libraries into the virtualenv... & pip install -r requirements.txt & echo. & echo Ready. Now you can run scripts. & echo ============================== EXAMPLES: =================================== & echo 	python dataset_magnatagatune.py & echo 	python dataset_lastfm.py --download_spotify_preview & echo 	python dataset_lastfm.py --convert_to_wav & echo 	python dataset_lastfm.py --compute_melgram & echo 	python neural_networks.py --architecture crnn4 --dataset lastfm & echo 	cd application ^& python application.py ^& cd .. & echo ============================================================================ & echo. & cmd /k