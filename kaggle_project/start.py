#%%
import pandas as pd 
import altair as alt 
import numpy as np 

from sklearn.model_selection import train_test_split as tts

# %%
articles = pd.read_csv('articles.csv')
customers = pd.read_csv('customers.csv')
sample = pd.read_csv('sample_submission.csv')
train = pd.read_csv('transactions_train.csv')

#%%
train['t_dat'] = pd.to_datetime(train.t_dat,infer_datetime_format=True)
#%% Create our own validation set
valid = train.loc[ train.t_dat >= pd.to_datetime('2020-09-16') ]
valid = valid.groupby('customer_id').article_id.apply(list).reset_index()
valid = valid.rename({'article_id':'prediction'},axis=1)
valid['prediction'] =\
    valid.prediction.apply(lambda x: ' '.join(['0'+str(k) for k in x]))

valid.to_csv('validation.csv')

#%%
id_to_index_dict = dict(zip(customers["customer_id"], customers.index))
index_to_id_dict = dict(zip(customers.index, customers["customer_id"]))

# for memory efficiency
train["customer_id"] = train["customer_id"].map(id_to_index_dict)

# for switching back for submission
sub["customer_id"] = sub["customer_id"].map(index_to_id_dict)




#%%




#%%










#%%
#%%
# %%
q = '''
select c.customer_id,
       FN,
       Active,
       club_member_status,
       fashion_news_frequency,
       age,
       postal_code,
       t_dat,
       price,
       sales_channel_id,
       a.article_id,
       product_code,
product_type_no,
graphical_appearance_no,
colour_group_code,
perceived_colour_value_id,
perceived_colour_master_id,
department_no,
index_code,
index_group_no,
section_no,
garment_group_no,
from customers c
join train t on c.customer_id = t.customer_id
join articles a on t.article_id = t.article_id
'''
trans = pysqldf(q)
# %%
