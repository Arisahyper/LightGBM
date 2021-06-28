import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


def main():
  titanic = sns.load_dataset("titanic")
  df = pd.DataFrame(titanic)
  df.to_csv("titanic_to_csv.csv")
  df = pd.read_csv("titanic_to_csv.csv", index_col=0)

  df['age'].fillna(df.age.median(), inplace=True)
  df.dropna(how="any", axis = 0, inplace=True)

  for column in ['sex','embarked', 'class', 'who', 'adult_male']:
      le = LabelEncoder()
      le.fit(df[column])
      df[column] = le.transform(df[column])

  x = pd.DataFrame(df, columns=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'class', 'who', 'adult_male'])
  y = df["survived"]

  x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)

  lgb_train = lgb.Dataset(x_train, y_train)
  lgb_val = lgb.Dataset(x_val, y_val, reference = lgb_train)

  # params = {
  #     'boosting_type': 'gbdt',
  #     "objective" : "binary",
  #     "metric":"binary_logloss",
  #     "num_iterations" : 100,
  #     "max_depth" : 2,
  # }

  params_v2 = {
      "objective" : "multiclass",
      "num_classes" : 2,
      "max_depth" : 2,
      "num_iterations" : 200,
      "num_reaves" : 30,
  }

  model = lgb.train(params_v2, lgb_train, valid_sets = lgb_val)

  y_pred = model.predict(x_val)
  y_pred_max = np.argmax(y_pred, axis = 1)

  accuracy = sum(y_val == y_pred_max) / (len(y_val))
  print(accuracy)

if __name__ == "__main__":
  main()
