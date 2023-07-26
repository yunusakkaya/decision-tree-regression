import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("decision-tree-regression-dataset.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)
#%%
"""from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)

tree_reg.predict(5,5)
x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=tree_reg.predict(x)
#%%
plt.scatter(x,y_head,color="red")
plt.plot(x_,y_head,color="blue")
plt.xlabel("tırubun level")
plt.ylabel("fiyat")
plt.show()"""
#%%
"""random forest"""

"""from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)
print("7.8:",rf.predict(7.8))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tirubun")
plt.ylabel("fiyat")
plt.show()"""
#%%
"""from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x,y)

y_head = rf.predict(x)"""
#%%
"""from sklearn.metrics import r2_score
print("r_score:", r2_score(y,y_head))"""




























