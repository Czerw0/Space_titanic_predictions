from ml_model import *
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

df_test1 = pd.read_csv(r"C:\Users\karol\OneDrive\SWPS\semestr 4\wprowadzenie do SI\space_titanic_test.csv")
df_test = df_test1.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

print(df_test.head())
print("\n")
print(df_test.describe().T)
print("\n")
df_test.info()
print("\n")

#missing values
#numerical
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(df_test[['Age']])
df_test[['Age']] = imputer.transform(df_test[['Age']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(df_test[['RoomService']])
df_test[['RoomService']] = imputer.transform(df_test[['RoomService']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(df_test[['FoodCourt']])
df_test[['FoodCourt']] = imputer.transform(df_test[['FoodCourt']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(df_test[['ShoppingMall']])
df_test[['ShoppingMall']] = imputer.transform(df_test[['ShoppingMall']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(df_test[['Spa']])
df_test[['Spa']] = imputer.transform(df_test[['Spa']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(df_test[['VRDeck']])
df_test[['VRDeck']] = imputer.transform(df_test[['VRDeck']])

#categorial
imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(df_test[['HomePlanet']])
df_test[['HomePlanet']] = imputerE.transform(df_test[['HomePlanet']])

imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(df_test[['CryoSleep']])
df_test[['CryoSleep']] = imputerE.transform(df_test[['CryoSleep']])

imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(df_test[['Destination']])
df_test[['Destination']] = imputerE.transform(df_test[['Destination']])

imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(df_test[['VIP']])
df_test[['VIP']] = imputerE.transform(df_test[['VIP']])

#one hot encoding
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_test[['HomePlanet']])
transformed_home = enc.transform(df_test[['HomePlanet']]).toarray()
home_df = pd.DataFrame(transformed_home,columns=enc.get_feature_names_out(["HomePlanet"]))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_test[['CryoSleep']])
transformed_cs = enc.transform(df_test[['CryoSleep']]).toarray()
cs_df = pd.DataFrame(transformed_cs,columns=enc.get_feature_names_out(["CryoSleep"]))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_test[['Destination']])
transformed_dest = enc.transform(df_test[['Destination']]).toarray()
dest_df = pd.DataFrame(transformed_dest,columns=enc.get_feature_names_out(["Destination"]))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_test[['VIP']])
transformed_vip = enc.transform(df_test[['VIP']]).toarray()
vip_df = pd.DataFrame(transformed_vip,columns=enc.get_feature_names_out(["VIP"]))

#final DF for test
X_transformed2 = df_test.copy()
X_test2 = X_transformed2.drop(['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], axis=1)
X_test2 = pd.concat([X_test2, home_df, cs_df, dest_df, vip_df], axis=1)

#Scaling
mm_age = MinMaxScaler()
age_minmax = mm_age.fit_transform(X_test2[['Age']])
X_test2['Age'] = age_minmax

mm_rs = MinMaxScaler()
rs_minmax = mm_rs.fit_transform(X_test2[['RoomService']])
X_test2['RoomService'] = rs_minmax

mm_fc = MinMaxScaler()
fc_minmax = mm_fc.fit_transform(X_test2[['FoodCourt']])
X_test2['FoodCourt'] = fc_minmax

mm_shop = MinMaxScaler()
shop_minmax = mm_shop.fit_transform(X_test2[['ShoppingMall']])
X_test2['ShoppingMall'] = shop_minmax

mm_spa = MinMaxScaler()
spa_minmax = mm_spa.fit_transform(X_test2[['Spa']])
X_test2['Spa'] = spa_minmax

mm_vr = MinMaxScaler()
vr_minmax = mm_vr.fit_transform(X_test2[['VRDeck']])
X_test2['VRDeck'] = vr_minmax

#Chacking
print(X_test2.head())
print("\n")
print(X_test2.isna().sum())
print("\n")
print(f"Shape of X_test2: {X_test2.shape}")