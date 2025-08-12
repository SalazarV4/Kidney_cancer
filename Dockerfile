FROM 3.13-slim-bullseye

RUN apt update -y && apt install awscli -y
WORKDIR /app

COPY . /app
RUN poetry install

CMD ["python", "app.py"]