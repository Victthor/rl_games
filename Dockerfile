FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

# RUN apt-get update
# RUN pip3 -q install pip --upgrade
# RUN pip install wandb
COPY requirements.txt requirements.txt
RUN python -m pip install -U pip && pip install -r requirements.txt

# RUN wandb login && echo "6fcd9405bc5a3376af3c153ee09e808cb84d384a"
CMD ["wandb login"] ["6fcd9405bc5a3376af3c153ee09e808cb84d384a"]
