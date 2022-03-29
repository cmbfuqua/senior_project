#%%
import pandas as pd
from pyrsistent import ny

# %%
dbscan_y = [47250,98000,150000,165000,172000,169000]
dbscan_x = [.02,.16,.54,.74,.74,.74]

iso_y = [181000,179000,170000,165000,164000,163000,159500,155000,150000,151000,150000,148000,145000,141000]
iso_x = [.95,.94,.92,.9,.88,.86,.84,.82,.8,.78,.76,.74,.72,.7]

lof_y = [215000,220000,222000,224000,225000,224000,222000,220000,219000,217000,215000,216000,215000]
lof_x = [.86,0.8575,0.855,0.8525,0.85,0.8475,0.845,0.8425,0.84,0.8375,0.835,0.8325,0.83]

normal_x = [0,1]
normal_y = [220000,220000]



# %%
dbscan = pd.DataFrame({'x':dbscan_x,'y':dbscan_y})
iso = pd.DataFrame({'x':iso_x,'y':iso_y})
lof = pd.DataFrame({'x':lof_x,'y':lof_y})
normal = pd.DataFrame({'x':normal_x,'y':normal_y})

normal['model'] = 'Initial RMSE'
dbscan['model'] = 'DBSCAN'
iso['model'] = 'Isolation Forest'
lof['model'] = 'Local Outlier Factor'
# %%
import altair as alt
db = alt.Chart(dbscan, title = 'DBSCAN Rows to RMSE').mark_line().encode(
    alt.X('x',title = 'Data Percent Remaining'),
    alt.Y('y',title = 'RMSE Score')
)
i = alt.Chart(iso,title = 'Isolation Forest Rows to RMSE').mark_line().encode(
    alt.X('x',title = 'Data Percent Remaining'),
    alt.Y('y',title = 'RMSE Score')
)

l = alt.Chart(lof,title = 'LOF Rows to RMSE').mark_line().encode(
    alt.X('x',title = 'Data Percent Remaining'),
    alt.Y('y',title = 'RMSE Score')
)
# %%
db
# %%
i
# %%
l
# %%
db.save('images/dbscan_rmse_remaining.png')
i.save('images/isoforest_rmse_remaining.png')
l.save('images/lof_rmse_remaining.png')
# %%
total = pd.concat([dbscan,iso,lof,normal])
# %%
combined = alt.Chart(total,title = 'RMSE to Rows Cross Model Comparison').mark_line().encode(
    alt.X('x',title = 'Data Percent Remaining'),
    alt.Y('y',title = 'RMSE Score'),
    alt.Color('model',title = 'Detection Type')
)
combined
# %%
combined.save('combined_rmse_remaining.png')
# %%
