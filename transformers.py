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

        # convert date to datetime
        X_["Date"] = pd.to_datetime(X_["Date"], infer_datetime_format=True, dayfirst=False)
        
        # creates new dataframe to store dates
        new = pd.DataFrame()

        new['day'] = X_['Date'].dt.day
        new['month'] = X_['Date'].dt.month
        new['year'] = X_['Date'].dt.year
        new['hour'] = X_['Date'].dt.hour
        new['day of the week'] = X_['Date'].dt.weekday
        
        return new
    
class TimeTransformer2(BaseEstimator, TransformerMixin):
    # transforms Date column into date components to be used in the model v2.0
    
    def __init__(self):
        
        return
    
    def fit(self, X, y=None):

        
        # always copy!
        X = X.copy()
        
        return self

    def transform(self, X, y=None):

        # always copy!
        X_ = X.copy()

        # convert date to datetime
        X_["Date"] = pd.to_datetime(X_["Date"], infer_datetime_format=True, dayfirst=False)
        
        # creates new dataframe to store dates
        new = pd.DataFrame()

        #new['day'] = X_['Date'].dt.day
        #new['month'] = X_['Date'].dt.month
        new['quarter'] = X_['Date'].dt.quarter
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
    
class lat_lon_imputer(BaseEstimator, TransformerMixin):
    # Imputes missing values in the Latitude and Longitude Columns
    
    def __init__(self):
        
        return
    
    def fit(self, X, y=None):
        # The fit function creates a dictionary that has the avg latitude and longitude for each station

        # always copy!
        df_impute = X[["Latitude", "Longitude", "station"]].copy()

        station_list = list(df_impute["station"].unique())

        self.station_dict = {}
        for station in station_list:

            if (station != 'south-yorkshire') & (station != 'nottinghamshire') :
                avg_lat = df_impute.loc[df_impute["station"]==station, "Latitude"].mean()
                avg_lon = df_impute.loc[df_impute["station"]==station, "Longitude"].mean()
            elif station == 'south-yorkshire':
                # since south-yorkshire has no latitude or longitude values in the data we are using the closest county (west-yorkshire)
                avg_lat = df_impute.loc[df_impute["station"]=='west-yorkshire', "Latitude"].mean()
                avg_lon = df_impute.loc[df_impute["station"]=='west-yorkshire', "Longitude"].mean()
            elif station == 'nottinghamshire':
                # since nottinghamshire has no latitude or longitude values in the data we are using the closest county (derbyshire)
                avg_lat = df_impute.loc[df_impute["station"]=='derbyshire', "Latitude"].mean()
                avg_lon = df_impute.loc[df_impute["station"]=='derbyshire', "Longitude"].mean()

            self.station_dict[station] = {'lat': avg_lat, 'lon': avg_lon}

        return self

    def transform(self, X, y=None):

        # always copy!
        df_impute = X[["Latitude", "Longitude", "station"]].copy()

        for station in self.station_dict:
            df_impute.loc[df_impute["station"]==station, "Latitude"] =  df_impute.loc[df_impute["station"]==station, "Latitude"].fillna(value=self.station_dict[station]['lat'])
            df_impute.loc[df_impute["station"]==station, "Longitude"] =  df_impute.loc[df_impute["station"]==station, "Longitude"].fillna(value=self.station_dict[station]['lon'])
        
        # df_impute["Latitude"] = df_impute["Latitude"].fillna()

        return df_impute[["Latitude", "Longitude"]].copy()
    

class Group_Age_Range(BaseEstimator, TransformerMixin):
    # Groups Age Ranges 'under 10' and '10-17' into the new category 'under 17'
    
    def __init__(self):
        
        return
    
    def fit(self, X, y=None):

        # always copy!
        X = X.copy()
        
        return self

    def transform(self, X, y=None):

        # always copy!
        
        X_ = X.copy()

        X_['Age range'] = X_['Age range'].astype('object')
        X_.loc[(X_['Age range'] == 'under 10') | (X_['Age range'] == '10-17'), 'Age range'] = 'under 17'
        X_['Age range'] = X_['Age range'].astype('category').cat.as_ordered().cat.reorder_categories(['under 17', '18-24', '25-34', 'over 34'], ordered=True)
        
        # df_impute["Latitude"] = df_impute["Latitude"].fillna()

        return X_.copy()


class Group_Ethnicity(BaseEstimator, TransformerMixin):
    # Groups Mixed ethnicity into Other
    
    def __init__(self):
        
        return
    
    def fit(self, X, y=None):

        # always copy!
        X = X.copy()
        
        return self

    def transform(self, X, y=None):

        # always copy!
        X_ = X.copy()

        X_.loc[(X_['Officer-defined ethnicity'] == 'Mixed'), 'Officer-defined ethnicity'] = 'Other'

        
        # df_impute["Latitude"] = df_impute["Latitude"].fillna()

        return X_.copy()

    
    