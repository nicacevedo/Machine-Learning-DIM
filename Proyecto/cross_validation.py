#%%
from numpy.lib.index_tricks import _fill_diagonal_dispatcher
import yfinance as yf

import pandas as pd
import numpy as np

from datetime import date, timedelta
from sklearn.impute import KNNImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import  SVC 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report



def list_diff(l1, l2):
    return [e for e in l1 if e not in l2]

# Clase de información de Stocks
class StocksData:

    def __init__(self, tickers, initial_date, last_date):
        self.tickers = tickers
        self.initial_date = initial_date
        self.last_date = last_date
        self.data = None
        self.volume_data = None

    # Get atribute
    def get_data(self):
        return self.data

    # Set atribute
    def set_data(self, new_data):
        self.data = new_data
        return self

    def get_volume_data(self):
        return self.volume_data

    def set_volume_data(self, new_data):
        self.volume_date = new_data
        return self

    def download_data(self, features=['Close'], volume=False):
        # Adjust features format (.SN)
        tickers_adj = ['{}.SN'.format(ticker.replace(' ','-')) for ticker in self.tickers]

        # Download data
        data = yf.download(tickers_adj, start=self.initial_date, end=self.last_date)

        if volume:
            # Drop columns 
            all_features = ['Open','High','Low', 'Adj Close', 'Volume', 'Close']
            data_volume = data['Volume']
            data.drop(columns=list_diff(all_features, features), inplace=True)

            # Change format again (.SN)
            data.columns = [tup[1][:-3] for tup in data.columns]
            data_volume.columns = [tup[1][:-3] for tup in data_volume.columns]

            # Data to atribute
            self.set_data(data)
            self.set_volume_data(data_volume)
        else:

            # Drop columns 
            all_features = ['Open','High','Low', 'Adj Close', 'Volume', 'Close']
            data.drop(columns=list_diff(all_features, features), inplace=True)

            # Change format again (.SN)
            data.columns = [tup[1][:-3] for tup in data.columns]

            # Data to atribute
            self.set_data(data)
        return self



    def clean_data(self):
        df = self.data.copy() # Copy

        # Drop días con demasiados NA's en el mercado (con 40% de los stocks)
        df.drop(df[df.transpose().isna().sum()>0.4*df.shape[1]].index, axis=0, inplace=True)

        # Drop de Stocks con más NA's de los que conviene predecir (5% de los días)
        df = df.drop(df.loc[:,df.isna().sum() > 0.05*df.shape[0]].columns, axis=1).copy()

        # Si aún quedan NA, se imputa
        if df.isna().sum().sum() > 0:
            # Impute with KNN prediction
            imputed = KNNImputer(n_neighbors=5, weights='distance').fit_transform(df)
            df = pd.DataFrame(imputed, columns=df.columns, index=df.index)

        self.set_data(df) # Change data
        return self

    def get_features(self):
        df = pd.DataFrame(index=self.data.transpose().index) # Final dataframe

        # 1. Mean of daily returns (% changes in price)
        price = self.data.iloc[1:,:]
        price_lag = self.data.iloc[:-1,:].to_numpy()
        # print('Features')
        # print(self.data)
        # print(price)
        # print(price_lag)
        daily_returns = ( price - price_lag )/price_lag
        # print(daily_returns)
        df['Return'] = np.mean( daily_returns )

        # 2. Volatility (std % changes in price)
        df['Volat'] = np.std( daily_returns )

        # 3. Range Normalized (How many stds on the range of % changes)
        df['RangeNorm'] = ( np.max(daily_returns) - np.min(daily_returns) ) / df['Volat']
        df.loc[ df['RangeNorm'].isna() , 'RangeNorm'] = 0 # Fill NA's when no variation
        
        # 4. MinMax Range (Ratio to know if Max is higher or Min is lower ) 
        max_diff = np.abs(np.max(daily_returns) - df['Return'])
        min_diff = np.abs(df['Return'] - np.min(daily_returns))
        df['MinMaxRatio'] = (max_diff - min_diff) / (df['Volat'] * df['RangeNorm'])
        df.loc[ df['MinMaxRatio'].isna() , 'MinMaxRatio'] = 0 # Fill NA's when no variation
        # df.loc[ df['MinMaxRatio'].isin([-np.inf, np.inf]) , 'MinMaxRatio'] = 0 # Fill NA's when no variation

        # 5. % of rises
        df['UpDownRatio'] = np.sum(daily_returns > 0) / self.data.shape[0]

        # 6. Sharpe ratio (taking risk-free as the mean of returns)
        df['Sharpe'] = ( df['Return'] - df['Return'].mean() )/df['Volat'].to_numpy()
        df.loc[ df['Sharpe'].isin([-np.inf, np.inf]) , 'Sharpe'] = 0 # Stock becomes risk-free

        # 7. 
        
        return df

    def get_target(self, threshold):
        df = pd.DataFrame(index=self.data.transpose().index) # Final serie

        # For now: target is nor the max difference between top peak and first price 
        # of the horizon surpasses thethreshold or not
        df['Target'] = ( self.data.max() - self.data.iloc[0,:] ) / self.data.iloc[0,:]
        df.loc[df['Target'] >= threshold, 'Target'] = 1
        df.loc[df['Target'] <  threshold, 'Target'] = 0
        return df['Target']

# Cross Validation temporal de stocks
class StocksTimeCV:

    def __init__(self, tickers, horizon_size, n_train_sets, last_date=date.today()):
        assert n_train_sets > 1, "El número de set de entrenamientos debe ser al menos 2"
        self.tickers = tickers
        self.last_date = last_date
        self.horizon = horizon_size # Weeks for now
        self.trains  = n_train_sets

    def train_test_split(self):
        delta_horizon = timedelta(weeks=self.horizon)
        delta_day = timedelta(days=1)

        train_stocks = []   
        # Each train set
        for i in range(self.trains):
            sub_last_date  = self.last_date - delta_horizon*(1+i)
            sub_first_date = sub_last_date - delta_horizon
            sub_last_date  = sub_last_date - delta_day # avoid repeated days
            # Stock object
            stocks = StocksData(self.tickers, sub_first_date, sub_last_date).download_data(volume=False).clean_data()
            train_stocks.append(stocks) # Add to train sets

            # Update tickers
            if len(stocks.get_data().columns) < len(self.tickers):
                self.tickers = list(stocks.get_data().columns)

        # Add atribute
        self.train_stocks = train_stocks

        # Same for test set
        first_date = self.last_date - delta_horizon
        stocks = StocksData(self.tickers, first_date, self.last_date).download_data(volume=False).clean_data()
        self.test_stocks = stocks

        # Update tickers
        if len(stocks.get_data().columns) < len(self.tickers):
            self.tickers = list(stocks.get_data().columns)

        # Last filter of new tickers in train
        for i in range(self.trains):
            new_data = self.train_stocks[i].get_data().loc[:,self.tickers]
            self.train_stocks[i].set_data(new_data) 

        return self

    def fit_i(self, i_train_set): # fit the i-th training set
        assert i_train_set > 0, "El índice debe ser mayor que 0"
        X = self.train_stocks[i_train_set].get_features()
        # print(X.isna().sum())
        # print(X.isin([-np.inf, np.inf]).sum())
        y = self.train_stocks[i_train_set-1].get_target(self.threshold)
        # # Scaler
        # X_scaled = MinMaxScaler().fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self,threshold, model):
        self.threshold = threshold
        self.model = model
        # Test sets
        X_test = self.test_stocks.get_features()
        y_test = self.test_stocks.get_target(self.threshold)

        # print(y_test)

        for i in range(1,self.trains):
            # Fit
            self.fit_i(i)
            y_pred = self.model.predict(X_test)
            print('-'*50)
            print(f'Métricas obtenidas para: {self.model}')
            print('-'*50)
            print(classification_report(y_test, y_pred, zero_division=0))



    
if __name__ == '__main__':
    
    # Acciones Chile (Banchile)
    tickers = pd.read_csv('./Datasets/acciones_chile.csv')['Acciones'].to_list() #['Symbol'].to_list()


    # Input
    weeks_horizon = 4   # delta t
    n_train_sets = 5    
    threshold = 0.04    # 
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    # model = SVC(kernel='linear')

    cv = StocksTimeCV(tickers, weeks_horizon, n_train_sets).train_test_split()
    cv.predict(threshold, model)#.predict()

    # tamaño de horizonte temporal a predecir (en semanas de momento)
    # numero de horizontes a tomar como entrenamiento
    # última fecha a tomar


# %%
