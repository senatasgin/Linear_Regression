# import library
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv("mart_ayi_sicaklik_.csv")

# plot data (verileri grafikte çizdirme)
plt.scatter(df.yillar,df.sicaklik)
plt.xlabel("Yıllar")
plt.ylabel("Sıcaklık")
plt.show()

#%% linear regression
# sklearn library (sklearn kütüphanesinin yüklenmesi)
from sklearn.linear_model import LinearRegression
x = df.yillar.values.reshape(-1,1)
y = df.sicaklik.values.reshape(-1,1)
# linear regression model (lineer regresyon modeli)
linear_reg = LinearRegression()
linear_reg.fit(x,y)

#%% prediction (tahmin etme)
print(linear_reg.predict([[2022]]))

