import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
pd.set_option('display.max_columns', None)

import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from scipy import stats
import statsmodels.api as sm
from scipy.special import inv_boxcox
from scipy.stats import boxcox
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from itertools import combinations

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
import pickle

# Imports for data visualization
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
from matplotlib import dates as mpld
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse as root

#Import of Prophet Library
import numpy as np

from scipy import stats
import statsmodels.api as sm
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import add_changepoints_to_plot

def getData():
    df = pd.read_excel('./monthly_index.xls')
    df.drop(['COMM_CODE', 'COMM_WT'], inplace=True, axis=1)
    reshaped_df = df.melt(id_vars=['COMM_NAME'],var_name='Month-Year',value_name='WPI')

    return reshaped_df

def commodity(df: pd.DataFrame):
    name = list(df['COMM_NAME'].unique())
    return name

def filterData(c_name,df):
    targetVariable = 'WPI'
    reshaped_df = df[df['COMM_NAME'] == c_name]
    reshaped_df['Month-Year'] = reshaped_df['Month-Year'].str.replace('INDX', '')
    time_series = pd.date_range(start='04/30/2012', end='12/31/2020', freq='M')
    duration = pd.DataFrame(data={"TimeSeries": time_series}, index=time_series)
    reshaped_df['TimeSeries'] = duration.index

    reshaped_df.index = reshaped_df['TimeSeries']
    skipFrom = '2013-01-01'
    reshaped_df_new = reshaped_df.loc[skipFrom:]
    product_data = reshaped_df
    # REsmapling the data
    product_data_year = product_data.resample('M').agg(
        {
            targetVariable: 'mean'

        })
    product_data_year['TimeSeries'] = product_data_year.index
    return product_data_year

def make_comparison_dataframe(historical, forecaste):
    return forecaste.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))


def modelBuildingStart(name, df):
    df = pd.read_excel('./monthly_index.xls')
    targetVariable = 'WPI'
    comm1 = df[df['COMM_NAME'] == name]
    # Dropping the Column
    comm1.drop(['COMM_CODE', 'COMM_WT'], inplace=True, axis=1)
    comm1.head(5)
    reshaped_df = comm1.melt(id_vars=['COMM_NAME'], var_name='Month-Year', value_name='WPI')
    reshaped_df.head(10)
    reshaped_df['Month-Year'] = reshaped_df['Month-Year'].str.replace('INDX', '')
    time_series = pd.date_range(start='04/30/2012', end='12/31/2020', freq='M')
    duration = pd.DataFrame(data={"TimeSeries": time_series}, index=time_series)
    reshaped_df['TimeSeries'] = duration.index

    reshaped_df.index = reshaped_df['TimeSeries']
    skipFrom = '2013-01-01'
    reshaped_df_new = reshaped_df.loc[skipFrom:]
    product_data = reshaped_df
    # REsmapling the data
    product_data_year = product_data.resample('M').agg(
        {
            targetVariable: 'mean'

        })
    product_data_year.head(2)
    product_data_year.tail(2)
    model_building_df = product_data_year
    model_building_df['TimeSeries'] = model_building_df.index
    # Model building
    # Choose prediction step
    prediction_size = 0  # Test Data Size
    train_dataset = pd.DataFrame()
    train_dataset['ds'] = pd.to_datetime(model_building_df['TimeSeries'])
    train_dataset['y'] = model_building_df[targetVariable]
    # train_dataset['AvgtempC'] = product_data['AvgtempC']
    # train_dataset['PrecipMM'] = product_data['PrecipMM']

    train_df = train_dataset.iloc[:len(train_dataset) - prediction_size, :]
    test_df = train_dataset.iloc[len(train_dataset) - prediction_size:, :]
    pro_regressor = Prophet(growth='linear', daily_seasonality=False, weekly_seasonality=False,
                            yearly_seasonality=False,
                            seasonality_mode='multiplicative',
                            changepoint_prior_scale=0.05,
                            )
    pro_regressor.fit(train_df)
    future_data = pro_regressor.make_future_dataframe(periods=prediction_size,
                                                      freq='M')  # -------------------- Chooose Frequency
    # future_data['AvgtempC']=train_dataset['AvgtempC']
    # future_data['PrecipMM']=train_dataset['PrecipMM']
    # future_data['AvgtempC'] = future_data['AvgtempC'].replace(np.nan,25)
    # future_data['PrecipMM'] = future_data['PrecipMM'].replace(np.nan,0)

    # forecast_data = pro_regressor.predict(test_df)
    # fig = pro_regressor.plot(forecast_data);

    forecaste = pro_regressor.predict(future_data)
    fig = pro_regressor.plot(forecaste);
    a = add_changepoints_to_plot(fig.gca(), pro_regressor, forecaste)
    forecaste_horizon = 10
    future_data = pro_regressor.make_future_dataframe(periods=forecaste_horizon, freq='M')
    forecast_data = pro_regressor.predict(future_data)
    cmp_df = make_comparison_dataframe(train_dataset, forecast_data)
    cmp_df['TimeSeries'] = cmp_df.index
    cmp_df['TimeSeries'] = cmp_df['TimeSeries'].dt.strftime('%Y-%m-%d')
    fromDate = '2019-12-31'
    #cmp_df = cmp_df.loc[fromDate:]
    #dates = list(cmp_df['TimeSeries'])
    #cmp_df['y'] = cmp_df['y'].replace(np.nan, 0)
    #actual = list(cmp_df['y'].astype(int))
    #forecasted = list(cmp_df['yhat'].astype(int))
    cmp_df.index = cmp_df['TimeSeries']
    return cmp_df