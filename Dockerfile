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

# Copy the src directory so poetry can find your package
COPY src ./src

# Install Python dependencies, now poetry can find your package
RUN poetry install --no-interaction --no-ansi

# Copy the rest of your project files
COPY . .

CMD ["poetry", "run", "python", "app.py"]
