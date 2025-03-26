import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from space_titanic_eda import df_train
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

csv_url = "https://raw.githubusercontent.com/<Czerw0>/<Space_titanic_predictions>/main/space_titanic_train.csv"
try:
    df_train = pd.read_csv(csv_url)
    print("File successfully read from GitHub!")
except Exception as e:
    print(f"Error reading file from GitHub: {e}")

df_train = df_train.drop(['PassengerId', 'Name', 'Cabin'], axis=1)

# defining X and y
X = df_train.drop('Transported', axis=1)
y = df_train['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# imputing missing values

#numerical
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(X[['Age']])
X[['Age']] = imputer.transform(X[['Age']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(X[['RoomService']])
X[['RoomService']] = imputer.transform(X[['RoomService']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(X[['FoodCourt']])
X[['FoodCourt']] = imputer.transform(X[['FoodCourt']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(X[['ShoppingMall']])
X[['ShoppingMall']] = imputer.transform(X[['ShoppingMall']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(X[['Spa']])
X[['Spa']] = imputer.transform(X[['Spa']])

imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(X[['VRDeck']])
X[['VRDeck']] = imputer.transform(X[['VRDeck']])

#categorial
imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(X[['HomePlanet']])
X[['HomePlanet']] = imputerE.transform(X[['HomePlanet']])

imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(X[['CryoSleep']])
X[['CryoSleep']] = imputerE.transform(X[['CryoSleep']])

imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(X[['Destination']])
X[['Destination']] = imputerE.transform(X[['Destination']])

imputerE = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputerE = imputerE.fit(X[['VIP']])
X[['VIP']] = imputerE.transform(X[['VIP']])

#Encoding categorical data
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X[['HomePlanet']])
transformed_home = enc.transform(X[['HomePlanet']]).toarray()
home_df = pd.DataFrame(transformed_home,columns=enc.get_feature_names_out(["HomePlanet"]))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X[['CryoSleep']])
transformed_cs = enc.transform(X[['CryoSleep']]).toarray()
cs_df = pd.DataFrame(transformed_cs,columns=enc.get_feature_names_out(["CryoSleep"]))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X[['Destination']])
transformed_dest = enc.transform(X[['Destination']]).toarray()
dest_df = pd.DataFrame(transformed_dest,columns=enc.get_feature_names_out(["Destination"]))

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X[['VIP']])
transformed_vip = enc.transform(X[['VIP']]).toarray()
vip_df = pd.DataFrame(transformed_vip,columns=enc.get_feature_names_out(["VIP"]))

#final data_frame

X_transformed = X.copy()
X_transformed = X_transformed.drop(['HomePlanet', 'CryoSleep', 'Destination', 'VIP'], axis=1)
X_transformed = pd.concat([X_transformed, home_df, cs_df, dest_df, vip_df], axis=1)

#Scaling 
mm_age = MinMaxScaler()
age_minmax = mm_age.fit_transform(X_transformed[['Age']])
X_transformed['Age'] = age_minmax

mm_rs = MinMaxScaler()
rs_minmax = mm_rs.fit_transform(X_transformed[['RoomService']])
X_transformed['RoomService'] = rs_minmax

mm_fc = MinMaxScaler()
fc_minmax = mm_fc.fit_transform(X_transformed[['FoodCourt']])
X_transformed['FoodCourt'] = fc_minmax

mm_shop = MinMaxScaler()
shop_minmax = mm_shop.fit_transform(X_transformed[['ShoppingMall']])
X_transformed['ShoppingMall'] = shop_minmax

mm_spa = MinMaxScaler()
spa_minmax = mm_spa.fit_transform(X_transformed[['Spa']])
X_transformed['Spa'] = spa_minmax

mm_vr = MinMaxScaler()
vr_minmax = mm_vr.fit_transform(X_transformed[['VRDeck']])
X_transformed['VRDeck'] = vr_minmax

#changing X_transformed

print(X_transformed.head())
print("\n")
print(X_transformed.describe().T)
print("\n")
print(X_transformed.info())
print("\n")
print(X_transformed.isna().sum())
print("\n")
print("X_transformed shape: ")
print(X_transformed.shape)
print("y shape: ")
print(y.shape)