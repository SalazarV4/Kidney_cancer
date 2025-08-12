FROM python:3.13-slim-bullseye

# Install OS-level dependencies and awscli
RUN apt update -y && apt install -y curl awscli && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --no-interaction --no-ansi

# Copy the rest of the project
COPY . .

CMD ["poetry", "run", "python", "app.py"]
