import os
import zipfile
import gdown
from kidney_cancer import logger
from kidney_cancer.utils.common import get_size
from kidney_cancer.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        """
        Fetch data from the url
        """

        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file
            os.makedirs(self.config.root_dir, exist_ok=True)

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?export=download&id="
            if os.path.exists(self.config.local_data_file):
                logger.info("File %s already exists, skipping download.",self.config.local_data_file)
            else:
                logger.info("Downloading dataset from %s into file %s",dataset_url,zip_download_dir)
                gdown.download(prefix+file_id, zip_download_dir)

        except Exception as e:
            raise e

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        if os.path.exists(r"artifacts\data_ingestion\Kidney_dataset"):
            logger.info("Dataset already unzipped!")
        else:
            logger.info("Unzipping data.zip")
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
