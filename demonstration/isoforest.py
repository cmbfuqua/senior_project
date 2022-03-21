#%%
from venv import create
import pandas as pd 
import altair as alt

from pandas_profiling import ProfileReport as pr 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# %%
dat = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
# Top 3 features: ['sqft_living','grade','sqft_lot15']

#%% 
# get rid of non-essential columns
def clean_house(dat):
    data = dat.drop(columns = ['id','date','lat','long'])

    # Clean up the data
    # bin the years
    data.loc[data.yr_built < 1950, 'yr_built_bin'] = '<1950'
    data.loc[(data.yr_built > 1950) & (data.yr_built < 2000), 'yr_built_bin'] = '<1950<2000'
    data.loc[data.yr_built > 2000, 'yr_built_bin'] = '>2000'
    # Change yr_renovated to boolean
    data.loc[data.yr_renovated != 0, 'yr_renovated'] = 1
    data = data.drop(columns = 'yr_built')

    # 
    # Categorical values that need to be one-hot encoded
    # bedrooms, bathrooms, floors, view, condition
    cat_list = ['bedrooms','bathrooms','floors','view','condition','yr_built_bin']
    for col in cat_list:
        onehot =  pd.get_dummies(data[col],col, drop_first=True)
        data = pd.concat([data,onehot],axis = 1)
        data = data.drop(columns=col)
    
    # 
    # Numerical values that need to be scaled
    num_list = ['sqft_living','sqft_lot', 'sqft_above', 'sqft_basement',
                'sqft_living15', 'sqft_lot15','grade']
    for col in num_list:
        mm = MinMaxScaler(feature_range = (0,10))
        mm.fit(pd.DataFrame(data[col]))
        data[col+'_scaled'] = mm.transform(pd.DataFrame(data[col]))
        data = data.drop(columns = col)

    # 
    # drop extra columns
    data = data.drop(columns = 'zipcode')

    return data
#%%
# clean up all data
data_train = clean_house(dat)
# Top 3 features: ['sqft_living','grade','sqft_lot15']
#%%
alt.data_transformers.enable(max_rows = None)
# Create charts for presentation
sqft = alt.Chart(data_train, title = 'Sqft Living Correlation').mark_point().encode(
    alt.X('price',title = 'Price'),
    alt.Y('sqft_living_scaled', title = 'Square ft Living Space')
)
grd = alt.Chart(data_train, title = 'Grade Correlation').mark_point().encode(
    alt.X('price',title = 'Price'),
    alt.Y('grade_scaled', title = 'Grade of House')
)

lot = alt.Chart(data_train,title = 'Sqft Lot Correlation').mark_point().encode(
    alt.X('price', title = 'Price'),
    alt.Y('sqft_lot_scaled',title = 'Sqft Plot of Land')
    
)

normal_chart = sqft|grd|lot

#%%
##############################
# import new models
##############################
from sklearn.ensemble import IsolationForest as isof
import numpy as np
##################################
# clean up data with outlier detection model
##################################
detection_model = isof()
'''
When allowed to determine the contamination rate, it comes
out to be around .05. 
'''

detection_model.fit(data_train)

iso = detection_model.predict(data_train)

iso = pd.DataFrame(iso)
indicies = iso.loc[iso[0] == 1].index
data_clean = data_train.iloc[indicies]
#%%
###################################
# Create new charts
###################################
nsqft = alt.Chart(data_clean, title = 'Sqft Living Correlation').mark_point().encode(
    alt.X('price',title = 'Price'),
    alt.Y('sqft_living_scaled', title = 'Square ft Living Space')
)
ngrd = alt.Chart(data_clean, title = 'Grade Correlation').mark_point().encode(
    alt.X('price',title = 'Price'),
    alt.Y('grade_scaled', title = 'Grade of House')
)

nlot = alt.Chart(data_clean,title = 'Sqft Lot Correlation').mark_point().encode(
    alt.X('price', title = 'Price'),
    alt.Y('sqft_lot_scaled',title = 'Sqft Plot of Land')
    
)

new_chart = nsqft|ngrd|nlot
new_chart
#%%
# %% This cell runs the model
# create x & y values
'''
Base RMSE without detection is ~220,000
'''
x = data_clean.drop(columns = 'price')
y = data_clean.price

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .3, random_state=76)

# use default values for knn model
knn = KNeighborsRegressor()

# fit and predict
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

# %%
rmse = metrics.mean_squared_error(y_test,y_pred,squared=False).round(2)
mse = metrics.mean_squared_error(y_test,y_pred).round(2)
mae = metrics.mean_absolute_error(y_test,y_pred).round(2)

print('MSE from data (Sensitive to large prediction error): {}'.format(mse))
print('MAE from data (Treats all errors equally): {}'.format(mae))
print('RMSE from data (Actual Units Off): {}'.format(rmse))
print('Base RMSE 217,240')
normal_chart & new_chart 

#%%
def create_effective_charts(data_train,contam):
    from sklearn.ensemble import IsolationForest
    import numpy as np
    ##################################
    # clean up data with outlier detection model
    ##################################
    detection_model = IsolationForest(
        contamination = contam
    )

    detection_model.fit(data_train)

    iso = detection_model.predict(data_train)

    iso = pd.DataFrame(iso)
    indicies = iso.loc[iso[0] == 1].index
    data_clean = data_train.iloc[indicies]

    #This cell runs the model
    # create x & y values
    x = data_clean.drop(columns = 'price')
    y = data_clean.price

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .3, random_state=76)

    # use default values for knn model
    knn = KNeighborsRegressor()

    # fit and predict
    knn.fit(x_train,y_train)

    y_pred = knn.predict(x_test)

    # 
    rmse = metrics.mean_squared_error(y_test,y_pred,squared=False).round(2)

    return rmse,data_clean.shape[0]

# %%
import time
rmse = []
contaminate = []
shape = []
base = []
for contam in range(5,35,5):
    contam = contam/100
    tic = time.perf_counter() # start timer
    rmse_loop,dat_shape = create_effective_charts(data_train,contam)
    # append values to lists
    contaminate.append(contam)
    rmse.append(rmse_loop)
    shape.append(dat_shape)
    base.append(20000)
    toc = time.perf_counter() # end timer
    print('contam: {}  shape: {}  time: {}\n\n'.format(contam,dat_shape,round(toc - tic,2)))
df_results = pd.DataFrame(
    {'rmse_values':rmse,
    'contam':contaminate,
    'remaining_rows':shape,
    'start_rows':base}
)
# %%
chart1 = alt.Chart(df_results).mark_line().encode(
    alt.X('contam',title = 'Contamination %'),
    alt.Y('rmse_values')
)
chart1

# %%
chart1.save('isoforest_rmse_curve.png')
# %%
from sklearn.ensemble import IsolationForest
detection_model = IsolationForest(
        contamination = .15
    )

detection_model.fit(data_train)

iso = detection_model.predict(data_train)

iso = pd.DataFrame(iso)
indicies = iso.loc[iso[0] == 1].index
data_clean = data_train.iloc[indicies]

#This cell runs the model
# create x & y values
x = data_clean.drop(columns = 'price')
y = data_clean.price

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .3, random_state=76)

# use default values for knn model
knn = KNeighborsRegressor()

# fit and predict
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

# 
rmse = metrics.mean_squared_error(y_test,y_pred,squared=False).round(2)
###################################
# Create new charts
###################################
nsqft = alt.Chart(data_clean, title = 'Sqft Living Correlation').mark_point().encode(
    alt.X('price',title = 'Price'),
    alt.Y('sqft_living_scaled', title = 'Square ft Living Space')
)
ngrd = alt.Chart(data_clean, title = 'Grade Correlation').mark_point().encode(
    alt.X('price',title = 'Price'),
    alt.Y('grade_scaled', title = 'Grade of House')
)

nlot = alt.Chart(data_clean,title = 'Sqft Lot Correlation').mark_point().encode(
    alt.X('price', title = 'Price'),
    alt.Y('sqft_lot_scaled',title = 'Sqft Plot of Land')
    
)

new_chart = nsqft|ngrd|nlot
new_chart.save('isolationForest_chart.png')
new_chart
# %%
