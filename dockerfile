FROM python:3.6-slim

WORKDIR /app

COPY . /app

RUN python -m pip install -U matplotlib==3.1.2
RUN pip install -r requirements.txt