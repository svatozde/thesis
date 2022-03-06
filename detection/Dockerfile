FROM python:3.8-slim-buster
COPY . /detection
WORKDIR detection
EXPOSE 5000
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --reinstall build-essential \
    && apt-get install -y gcc \
    && apt-get install -y git \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && pip3 install --upgrade cython \
    && pip3 install --upgrade decorator \
    && pip3 install git+https://github.com/pattern-inc/cynetworkx.git \
    && pip3 install numpy \
    && pip3 install .
ENTRYPOINT ["python", "server/detection_server.py"]
