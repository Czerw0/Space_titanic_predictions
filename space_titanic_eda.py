import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"C:\Users\karol\OneDrive\SWPS\semestr 4\wprowadzenie do SI\space_titanic_train.csv"
df_train = pd.read_csv(file_path)

print(df_train.head())
print("\n")
print(df_train.describe().T)
print("\n")
df_train.info()
print("\n")

# FacetGrid histogram
g = sns.FacetGrid(df_train, col='Transported')
g.map(plt.hist, 'Age', bins=10)
plt.show()

# Boxplots
fig, axs = plt.subplots(2, 3, figsize=(20, 10))
fig.suptitle("Boxplots of Different Features", fontsize=18)

sns.boxplot(x=df_train['Age'], ax=axs[0, 0], color='lightblue')
axs[0, 0].set_title("Age Distribution")

sns.boxplot(x=df_train['RoomService'], ax=axs[0, 1], color='lightcoral')
axs[0, 1].set_title("Room Service Spending")

sns.boxplot(x=df_train['FoodCourt'], ax=axs[0, 2], color='lightgreen')
axs[0, 2].set_title("Food Court Spending")

sns.boxplot(x=df_train['ShoppingMall'], ax=axs[1, 0], color='gold')
axs[1, 0].set_title("Shopping Mall Spending")

sns.boxplot(x=df_train['VRDeck'], ax=axs[1, 1], color='purple')
axs[1, 1].set_title("VR Deck Spending")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
plt.show()


#missing values
plt.figure(figsize=(10,5))
sns.heatmap(df_train.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

#missing values 2 
print("Missing values in df_train: ")
print(df_train.isnull().sum())

