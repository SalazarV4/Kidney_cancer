import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

PROJECT_NAME = "kidney_cancer"

list_of_paths = [
    ".github/workflows/.gitkeep",
    f"src/{PROJECT_NAME}/components/__init__.py",
    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/config/__init__.py",
    f"src/{PROJECT_NAME}/config/configuration.py",
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    f"src/{PROJECT_NAME}/entity/__init__.py",
    f"src/{PROJECT_NAME}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "research/trials.ipynb",
    "templates/index.html"
]

for path in list_of_paths:
    filepath = Path(path)
    filedir, filename = os.path.split(filepath)

    if (filedir != "") and (not os.path.exists(filedir)):
        os.makedirs(filedir, exist_ok=True)
        logging.info("Creating Directory: %s", filedir)
    else:
        _, directory = os.path.split(filedir)
        if filedir != '':
            logging.info("%s directory already exists", directory)

    if not os.path.exists(filepath):
        with open(filepath, "w", encoding='utf-8') as f:
            logging.info("Creating Empty file: %s",filepath)

    else:
        logging.info("%s already exists", filename)