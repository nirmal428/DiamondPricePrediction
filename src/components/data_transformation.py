from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


## data transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')


## data ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config =DataTransformationconfig()


    def get_data_transformation_object(self):
         try:
            logging.info("Data Transformation initiated")
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Pipeline Initiated')

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalEncoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                    ('scaler', StandardScaler())
                ]
            )

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])

            logging.info('Pipeline Complete ')
            return preprocessor

         except Exception as e:
            logging.error(" Error in data transformation")
            raise CustomException(e, sys)
         
    def initiate_data_transformation(self,train_path,test_path):
     try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info('Read train and test data completed')
        logging.info(f'Train Dataframe Head: \
        n{train_df.head().to_string()}') 
        logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')
        logging.info('Obtaining preprocessing object')
        preprocessing_obj = self.get_data_transformation_object()     

        target_column_name='price'
        drop_columns=[target_column_name,'id']

        input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
        target_feature_train_df=train_df[target_column_name]

        input_feature_test_df=train_df.drop(columns=drop_columns,axis=1)
        target_feature_test_df=train_df[target_column_name]


        ##apply transformation
        input_feature_train_df_arr=preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_df_arr=preprocessing_obj.fit_transform(input_feature_test_df)

        logging.info("Applying preprocessing object on training and testing datasets. ")

        train_arr=np.c_[input_feature_train_df_arr,np.array(target_feature_train_df)]
        test_arr=np.c_[input_feature_test_df_arr,np.array(target_feature_test_df)]

        save_object(
           file_path=self.data_transformation_config.preprocessor_obj_file_path,
           obj=preprocessing_obj
        )

        logging.info('Preprocessor picke in created and save')

        return(
           train_arr,
           test_arr,
           self.data_transformation_config.preprocessor_obj_file_path
        )
     except Exception as e:
       logging.info("Exception accured in the initiate datatransformation")

       raise CustomException(e,sys)