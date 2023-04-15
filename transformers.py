from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TimeTransformer(BaseEstimator, TransformerMixin):
    # transforms Date column into date components to be used in the model
    
    def __init__(self):
        
        return
    
    def fit(self, X, y=None):

        
        # always copy!
        X = X.copy()
        
        return self

    def transform(self, X, y=None):

        # always copy!
        X_ = X.copy()

        new = pd.DataFrame()
        new['day'] = X_['Date'].dt.day
        new['month'] = X_['Date'].dt.month
        new['year'] = X_['Date'].dt.year
        new['hour'] = X_['Date'].dt.hour
        new['day of the week'] = X_['Date'].dt.weekday
        
        return new
    


class BoolTransformer(BaseEstimator, TransformerMixin):
    # Fills missing values with False and converts boolean values to numeric
    
    def __init__(self):
        
        return
    
    def fit(self, X, y=None):

        
        # always copy!
        X = X.copy()
        
        return self

    def transform(self, X, y=None):

        # always copy!
        X_ = X.copy()

        X_['Part of a policing operation'] = X_['Part of a policing operation'].fillna(value=False)
        X_['Part of a policing operation'] = X_['Part of a policing operation'] * 1.0
        
        return X_