import pandas as pd
d0=pd.read_csv("submit.D32.alpha0.11.epoch3.csv")
d1=pd.read_csv("submit.D32.alpha0.1.epoch4.csv")
d2=pd.read_csv("submit.D32.alpha0.11.epoch3.csv")
d3=pd.read_csv("submit.D32.alpha0.125.epoch2.csv")
d4=pd.read_csv("submit.D32.alpha0.125.epoch4.csv")
d5=pd.read_csv("submit.D28.alpha0.11.epoch2.csv")
d6=pd.read_csv("submit.hour.D32.alpha0.11.epoch3.csv")

d=(d0+d1+d2+d3+d4+d5+d6)/7
d["Id"]=d0["Id"]

d.to_csv("submit.ave.7.csv",index=False)


