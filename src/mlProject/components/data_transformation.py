import os
from src.mlProject import logger
from src.mlProject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import pandas as pd

class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config


    ## Note: Different types of data transformations like scaler, PCA, for categorical data - apply encoder techniques


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # split the data into training and test sets with 75 and 25 percent.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir,"train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir,"test.csv"),index = False)

        logger.info("Dataset splitted into Train and Test")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)