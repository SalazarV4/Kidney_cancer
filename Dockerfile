FROM python:3.13.5-slim-buster

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN poetry install

CMD ["python", "app.py"]