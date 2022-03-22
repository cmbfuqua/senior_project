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
from sklearn.neighbors import LocalOutlierFactor as lof
import numpy as np
##################################
# clean up data with outlier detection model
##################################
detection_model = lof(
    n_neighbors=2
)

pred = pd.DataFrame(detection_model.fit_predict(data_train))

indicies = pred.loc[pred[0] == 1].index
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
Base RMSE without detection is ~220000
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
def create_effective_charts(data_train,neigh):
    from sklearn.neighbors import LocalOutlierFactor as lof
    import numpy as np
    ##################################
    # clean up data with outlier detection model
    ##################################
    detection_model = lof(
        n_neighbors=neigh
    )

    pred = pd.DataFrame(detection_model.fit_predict(data_train))

    indicies = pred.loc[pred[0] == 1].index
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
neighbors = []
shape = []
base = []
for neighbor in [4,5,6,7,8,9,10]:
    tic = time.perf_counter() # start timer
    rmse_loop,dat_shape = create_effective_charts(data_train,neighbor)
    # append values to lists
    neighbors.append(neighbor)
    rmse.append(rmse_loop)
    shape.append(dat_shape)
    base.append(20000)
    toc = time.perf_counter() # end timer
    print('neighbors: {}  time: {}\n\n'.format(neighbor,round(toc - tic,2)))
df_results = pd.DataFrame(
    {'rmse_values':rmse,
    'n_neighbors':neighbors,
    'remaining_rows':shape,
    'start_rows':base}
)
# %%
chart1 = alt.Chart(df_results,title = 'RMSE Curve').mark_line().encode(
    alt.X('n_neighbors'),
    alt.Y('rmse_values')
)
chart1
# %%
df_results['percent_remaining'] = df_results.remaining_rows/20000
chart2 = alt.Chart(df_results,title = '% Rows Remaining').mark_line().encode(
    alt.X('n_neighbors'),
    alt.Y('percent_remaining'),
    alt.Color('min_samples:O',sort='descending')
)
chart_fin = chart1 | chart2
# %%
chart_fin.save('lof_percent_remaining.png')
# %%
##################################
# clean up data with outlier detection model
##################################
# Insert the best eps and min_samples values found above

detection_model = lof(
    n_neighbors=10
)

pred = pd.DataFrame(detection_model.fit_predict(data_train))

indicies = pred.loc[pred[0] == 1].index
data_clean = data_train.iloc[indicies]

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
new_chart.save('lof_chart.png')
new_chart
# %%
