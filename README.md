# ATCC_Eyeudio


<!-- copied from https://github.com/momo7773/Eyeudio -->
conda create -n eyeudio_env_3.7.7 python=3.7.7
conda activate eyeudio_env_3.7.7

# For Lip Preprocessing/Deep Lip Reading
conda install -c conda-forge opencv=4.5.3 dlib=19.22.0 sk-video=1.1.10
python -m pip install tensorflow-gpu
python -m pip install numpy
python -m pip install av
python -m pip install editdistance
python -m pip install configargparse
python -m pip install six
python -m pip install moviepy==0.2.3.5
python -m pip install opencv-python
python -m pip install imageio-ffmpeg
python -m pip install tensorflow_addons

# For Kivy (GUI)
python -m pip install "kivy[base]" kivy_examples

# For Speech Recognition/Syntax Checker
python -m pip install SpeechRecognition
python -m pip install pyautogui
python -m pip install osascript
python -m pip install fuzzywuzzy
conda install -c conda-forge python-levenshtein
python -m pip install ./PyAudio-0.2.11-cp37-cp37m-win_amd64.whl

# For Eye Tracking
conda install pytorch torchvision torchaudio cpuonly -c pytorch
python -m pip install omegaconf
python -m pip install face-alignment
python -m pip install fvcore
python -m pip install pandas
python -m pip install pyyaml
python -m pip install scipy
python -m pip install tensorboardX
python -m pip install yacs
python -m pip install mediapipe
python -m pip install timm
python -m pip install playsound==1.2.2
# pip install -r requirements.txt