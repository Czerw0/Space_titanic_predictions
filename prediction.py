from ml_model import best_rf
from space_titanic_test_fe import df_test, X_test2, df_test1

#final prediction
predictions = best_rf.predict(X_test2)
df_test['predictions'] = predictions
df_test['PassengerId'] = df_test1['PassengerId']

#outcome 
print("\n")
print("Predictions: ")
print(df_test.head(20))

#saving file
try:
    df_test.to_csv('titanic_predictions.csv', index=False)
    print("\n")
    print("File saved as titanic_predictions.csv")
except:
    print("Error saving file")