#!/usr/bin/env python
# coding: utf-8

# In[1]:

from typing import Union
from sklearn.preprocessing import *
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
import pandas as pd
import numpy as np


# In[9]:


class DataSet(pd.DataFrame):
    
    def __init__(self, path, time_cols, target, multivariate=True):
        """
        Object inherits from pd DataFrame; preprocessing functionality added
        """
        super().__init__(data=pd.read_csv(f'data/{path}.csv', index_col=0))

        self.time_cols = time_cols
        self.target = target
        self.multivariate = multivariate

        self.X_transformer = None
        self.y_transformer = None

        self.X = {'train': None,
                  'val': None,
                  'test': None}
        self.y = {'train': None,
                  'val': None,
                  'test': None}
        
    def handle_na(self, data):
        """
        Handles missing values
        """

        # Interpolate NA's where possible, keeping sequence in mind
        data.interpolate(limit_direction='forward', limit_area='inside', inplace=True)

        # Drop rest of rows with NA's
        data.dropna(inplace=True)
        
        X = data.drop(self.target, axis=1)
        y = data[self.target]
        
        return X, y


    def preprocess(self, val=True):
        """
        Function to take care of preprocessing steps
        """
        
        # Parse time data and split features and target
        if self.index.name not in self.time_cols:
            self['date'] = pd.to_datetime(self[self.time_cols])
            self.drop(self.time_cols, axis=1, inplace=True)
            self.set_index('date', inplace=True)
        
        X = self.drop(self.target, axis=1)
        y = self[self.target]
        
        # First split data before doing anything else
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_val, y_val = None, None
        if val:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2/0.8, shuffle=False)
            X_val, y_val = self.handle_na(pd.concat([X_val, y_val], axis=1))

        # Impute NA's and delete rest of NA's
        X_train, y_train = self.handle_na(pd.concat([X_train, y_train], axis=1))
        X_test, y_test = self.handle_na(pd.concat([X_test, y_test], axis=1))
                
        # Normalize numerical values and one hot encode categorical values
        self.X_transformer = make_column_transformer(
           (MinMaxScaler(),
            make_column_selector(dtype_include=np.number)),  
           (OneHotEncoder(),
            make_column_selector(dtype_include=object))
        )
        self.y_transformer = MinMaxScaler()
        
        # Fit scalers on train data and transform rest of data as well
        self.X['train'] = self.X_transformer.fit_transform(X_train)
        self.X['val'] = self.X_transformer.transform(X_val)
        self.X['test'] = self.X_transformer.transform(X_test)
        
        self.y['train'] = self.y_transformer.fit_transform(np.array(y_train).reshape(-1, 1))
        self.y['val'] = self.y_transformer.transform(np.array(y_val).reshape(-1, 1))
        self.y['test'] = self.y_transformer.transform(np.array(y_test).reshape(-1, 1))
        
        # If univariate, X = y
        if not self.multivariate:
            self.X = self.y
            self.X_transformer = self.y_transformer

    def get_data(self, split: Union[str, list]):
        """
        Fetch data from dataset
        """

        # If you only want train data, only return that
        if isinstance(split, str):
            return self.X[split], self.y[split]

        return [(self.X[s], self.y[s]) for s in split]

    def get_transformer(self, x_y: str = 'y') -> MinMaxScaler:
        """
        Fetch transformer for inverse transform
        """

        if x_y == 'X':
            return self.X_transformer

        return self.y_transformer


    def get_loaders(self, split: str, length: int, batch_size: int):
        """
        Function to get TimeseriesGenerator object
        """

        generator = TimeseriesGenerator(data = np.squeeze(self.X[split]), targets = np.squeeze(self.y[split]),
                                        length = length, batch_size = batch_size)

        return generator