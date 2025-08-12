FROM python:3.13-slim-bullseye

RUN apt update -y && apt install -y curl awscli && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY . /app

RUN poetry install --no-interaction --no-ansi


CMD ["poetry", "run", "python", "app.py"]