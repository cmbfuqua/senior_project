#%%
import pandas as pd 
import altair as alt

from pandas_profiling import ProfileReport as pr 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

# %%
dat = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing.csv')
hold_out_x = pd.read_csv('https://raw.githubusercontent.com/byui-cse/cse450-course/master/data/housing_holdout_test.csv')
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
    # bedrooms, bathrooms, floors, view, condition, grade
    cat_list = ['bedrooms','bathrooms','floors','view','condition','grade','yr_built_bin']
    for col in cat_list:
        onehot =  pd.get_dummies(data[col],col, drop_first=True)
        data = pd.concat([data,onehot],axis = 1)
        data = data.drop(columns=col)
    
    # 
    # Numerical values that need to be scaled
    num_list = ['sqft_living','sqft_lot', 'sqft_above', 'sqft_basement',
                'sqft_living15', 'sqft_lot15']
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
hold_out_x = clean_house(hold_out_x)
# %% This cell runs the model
# create x & y values
x = data_train.drop(columns = 'price')
y = data_train.price

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .3, random_state=76)

# use default values for knn model
knn = KNeighborsRegressor()

# fit and predict
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

#%%
hold_out_y = pd.read_csv('housing_holdout_targets.csv')
hold_out_pred = knn.predict(hold_out_x)

# %%
###############################################################
# Everything is the same up until this point
# and until you see the next big block like this
# here we will clean the data with the Local Outlier Factor model
################################################################
#%%


#%%
################################################################
# Resume the standard procedure
################################################################
# create x & y values
x = data_train.drop(columns = 'price')
y = data_train.price

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= .3, random_state=76)

# use default values for knn model
knn = KNeighborsRegressor()

# fit and predict
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

#%%
hold_out_y = pd.read_csv('housing_holdout_targets.csv')
hold_out_pred = knn.predict(hold_out_x)

# %%
rmse = metrics.mean_squared_error(y_test,y_pred,squared=False).round(2)
mse = metrics.mean_squared_error(y_test,y_pred).round(2)
mae = metrics.mean_absolute_error(y_test,y_pred).round(2)

rmse1 = metrics.mean_squared_error(hold_out_y,hold_out_pred,squared=False).round(2)
mse1 = metrics.mean_squared_error(hold_out_y,hold_out_pred).round(2)
mae1 = metrics.mean_absolute_error(hold_out_y,hold_out_pred).round(2)

print('MSE from data (Sensitive to large prediction error): {}'.format(mse))
print('MAE from data (Treats all errors equally): {}'.format(mae))
print('RMSE from data (Actual Units Off): {}'.format(rmse))

print('MSE from hold_out (Sensitive to large prediction error): {}'.format(mse1))
print('MAE from hold_out (Treats all errors equally): {}'.format(mae1))
print('RMSE from hold_out (Actual Units Off): {}'.format(rmse1))

# %%
# check for overfitting and feature_importances
# %%
