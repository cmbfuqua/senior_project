#%%
import altair as alt
import pandas as pd 

#%% GLOBAL DETECTION
data = {
    'Y':[12, 15, 17, 13, 20, 19, 45],
    'X':[15, 18, 12, 19, 13, 17, 50]

}

data = pd.DataFrame(data)
chart1 = alt.Chart(data, title = 'Sample 1').mark_point().encode(
    alt.X('X')
)
chart2 = alt.Chart(data, title = 'Sample 2').mark_point().encode(
    alt.Y('Y')
)
chart1.save('images/global1.png')
chart2.save('images/global2.png')
#%% LOCAL DETECTION
x = []
y = []
for i in range(100):
    ys = (i-50)**2
    x.append(i)
    y.append(ys)

data = {
    'Y':y,
    'X':x

}

data = pd.DataFrame(data)
data['Y'][50] = 500
data['Y'][49] = 500
data['Y'][51] = 500
chart1 = alt.Chart(data, title = 'Sample 1').mark_point().encode(
    alt.X('X')
)

chart2 = alt.Chart(data, title = 'Sample 2').mark_point().encode(
    alt.Y('Y')
)

chart3 = alt.Chart(data, title = 'Sample 3').mark_line(point = True).encode(
    alt.X('X'),
    alt.Y('Y')
)

chart1.save('images/local1.png')
chart3.save('images/local3.png')
chart2.save('images/local2.png')
#%%