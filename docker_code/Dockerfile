# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim-buster

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Set current directory as ENV
ENV PATH=/app:$PATH

# Needed for tzdata
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Los_Angeles

# copy items
WORKDIR /app
COPY tester-R.py ./
COPY popular-videos.csv ./
COPY requirements.txt ./
ADD ytdriver ./ytdriver
ADD SAVED_MODELS ./SAVED_MODELS


# Install pre-dependencies
RUN apt update
RUN apt install -y wget g++ unzip xvfb firefox-esr

# install python dependencies
RUN pip install -r requirements.txt

# Install chrome
RUN wget http://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_91.0.4472.101-1_amd64.deb
RUN apt install -y ./google-chrome-stable_91.0.4472.101-1_amd64.deb

# Download geckodriver
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux32.tar.gz
RUN tar xvf geckodriver-v0.29.1-linux32.tar.gz

# Download chromedriver
RUN wget https://chromedriver.storage.googleapis.com/91.0.4472.101/chromedriver_linux64.zip
RUN unzip chromedriver_linux64.zip
