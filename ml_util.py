import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error


class ReadData(BaseEstimator, TransformerMixin):
    """A custorm transformer to read .csv data file"""
    def __init__(self):
        self.file_path = None

    def fit(self, X, y=None):
        self.file_path = X
        return self
    
    def transform(self, X, y=None):
        self.file_path = X
        data = pd.read_csv(self.file_path)
        return data
    

class CleanText(BaseEstimator, TransformerMixin):
    """A custorm transformer to clean text data"""
    def __init__(self):
        self.char_columns = None

    def fit(self, X, y=None):
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        self.char_columns = [i for i in X.columns if i not in numeric_columns]
        return self
    
    def transform(self, X, y=None):
        # X[:, self.char_columns] = X[:, self.char_columns].str.lower()
        # X[self.char_columns] = X[self.char_columns].applymap(lambda x: x.str.lower() if isinstance(X[self.char_columns], str) else x)
        X[self.char_columns] = X[self.char_columns].apply(lambda x: x.str.lower() if x.dtype=='object' else x)        
        return X


class DropMissing(BaseEstimator, TransformerMixin):
    """A custorm transformer to treat missing values"""
    def __init__(self, columns=None):
        self.columns = None

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        data = X.dropna(subset=self.columns)
        data.reset_index(inplace=True, drop=True)
        return data


class SeperateLabel(BaseEstimator, TransformerMixin):
    def __init__(self, label=None):
        self.label = label
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.label is None:
            return X
        else:
            y = X.loc[:, self.label]
            X = X.drop(columns=self.label)
            return X, y
    

class SplitData(BaseEstimator, TransformerMixin):
    """A custorm transformer to split data set into training and test sets"""
    def __init__(self, strat=True, cat=None):
        self.strat = strat
        self.cat = cat
        self.y = None
        self.cat_field = None
        self.spliter = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None

    def fit(self, X, y):
        self.y = y
        self.cat_field = X.loc[:, self.cat]
        return self

    def transform(self, X, y=None):
        if self.strat == True:
            self.spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  
            for train_index, test_index in self.spliter.split(X, self.cat_field):
                self.train_x = X.iloc[train_index, :]
                self.test_x = X.iloc[test_index, :]
                self.train_y = self.y.iloc[train_index]
                self.test_y = self.y.iloc[test_index]
        else:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(X, self.y, test_size=0.2, random_state=42)
        self.train_x = self.train_x.reset_index(drop=True)
        self.test_x = self.test_x.reset_index(drop=True)
        self.train_y = self.train_y.reset_index(drop=True)
        self.test_y = self.test_y.reset_index(drop=True)
        return self.train_x, self.test_x, self.train_y, self.test_y


class FillMissing(BaseEstimator, TransformerMixin):
    """A custorm transformer to fill missing numeric values"""
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = None
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.select_dtypes(include=[np.number]).columns
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X.loc[:, self.columns])
        return self
    
    def transform(self, X, y=None):
        data = self.imputer.transform(X.loc[:, self.columns])
        data = pd.DataFrame(data, columns=self.columns)
        data = pd.concat([data, X.drop(columns=self.columns)], axis=1)
        return data
    

class EncodeCategory(BaseEstimator, TransformerMixin):
    """A custorm transformer to encode categorical data"""
    def __init__(self, feature='ocean_proximity', one_hot=True):
        self.feature = feature
        self.one_hot = one_hot
        self.enc = None

    def fit(self, X, y=None):
        if self.one_hot == True:
            self.enc = OneHotEncoder(
                sparse_output=False, 
                categories=[['near bay', '<1h ocean', 'inland', 'near ocean', 'island']]
            )
            self.enc.fit(X[[self.feature]])
            return self
        else:
            self.enc = OrdinalEncoder(categories=[['island', 'near ocean', '<1h ocean', 'near bay', 'inland']])
            self.enc.fit(X[[self.feature]])
            return self
    
    def transform(self, X, y=None):
        feature_cat = X[self.feature].unique()
        if self.one_hot == True:
            encoded = self.enc.transform(X[[self.feature]])
            encoded_df = pd.DataFrame(encoded, columns=self.enc.get_feature_names_out([self.feature]))
            data = pd.concat([X.drop(columns=[self.feature]), encoded_df], axis=1)
            return data
        else:
            data = self.enc.transform(X[[self.feature]])
            X = X.drop(columns=[self.feature])
            X[self.feature] = data
            return X


class CreateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, create_features=True):
        self.create_features = create_features

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.create_features == True:
            X['population_per_households'] = X['population'] / X['households']
            X['population_per_total_rooms'] = X['population'] / X['total_rooms']
            X['median_income_per_households'] = X['median_income'] / X['households']
            X['total_bedrooms_per_total_rooms'] = X['total_bedrooms'] / X['total_rooms']
            return X
        else:
            return X


class ScaleFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, standard=True):
        self.standard = standard
        self.scaler = None
    
    def fit(self, X, y=None):
        if self.standard == True:
            self.scaler = StandardScaler()
            return self
        else:
            self.scaler = MinMaxScaler()
            return self
    
    def transform(self, X, y=None):
        data = self.scaler.fit_transform(X) 
        return data
  

def train_error(train_y, train_y_pred):
    """Returns training error score"""
    mse = mean_squared_error(train_y, train_y_pred)
    rmse = np.sqrt(mse) # np.sqrt(np.mean(np.power(train_y_pred-train_y, 2)))
    return {"Training_error": rmse}

def cv_error(model, train_x, train_y):
    """Returns cross-validation error scores"""
    cv_scores = cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv=5)
    rmse_scores = np.sqrt(-cv_scores)
    return {
        "Validation_errors": rmse_scores, 
        "Mean": rmse_scores.mean(), 
        "Standard deviation": rmse_scores.std()
    }

# Learnining curve
def plot_learning_curves(model, train_x, train_y):
    X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")