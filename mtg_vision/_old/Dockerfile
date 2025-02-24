
# COMMAND USED TO RUN ON LINUX
# docker build ./ -t mtg-net && nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --user $(id -u):$(id -g) -v "$HOME/downloads/datasets:/datasets/" -v "$HOME/downloads/tmp/pycharm_project_106:/workspace" mtg-net python train_tf_net.py

FROM nvcr.io/nvidia/tensorflow:19.04-py3

RUN apt update
RUN apt install -y libsm6 libxext6 libxrender-dev

RUN pip install numpy requests scikit-image scikit-learn pandas pillow pytest imageio pyyaml tqdm termcolor \
        jsonpickle mtgtools dash \
        opencv-contrib-python \
        keras
RUN pip install wandb
