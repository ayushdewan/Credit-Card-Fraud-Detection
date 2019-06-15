import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1: Data Collection
df = pd.read_csv("creditcard.csv")

print("Data read from file")

# 2: Feature Engineering
from sklearn.preprocessing import StandardScaler

df["normAmount"] = StandardScaler().fit_transform(np.array(df["Amount"]).reshape(-1,1))
df = df.drop(["Time", "Amount"], axis=1)

print("Features selected")

# 3: Training and Testing Split
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.33, random_state=42)
X_train = train.drop("Class", axis=1)
y_train = train["Class"]
X_test = test.drop("Class", axis=1)
y_test = test["Class"]

print("Training and Testing Sets Created")


# 4: Minority Class Oversampling
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print("Minority Class Oversampled")

# 5: Classification Model Bruteforce
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

models = [
    LinearSVC(random_state=0),
    LogisticRegression(random_state=0),
    DecisionTreeClassifier(random_state=0),
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
]
print("Models initialized")

models_df = pd.DataFrame(columns=["model_type", "score_type", "score"])

for clf in models:
    print("Model:", clf)

    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test)

    recall = recall_score(y_test, y_pred.round())
    print("Recall score: {0:0.3f}".format(recall))

    f1 = f1_score(y_test, y_pred.round())
    print("f1 score: {0:0.3f}".format(f1))

    model_type = model_type = type(clf).__name__

    models_df = models_df.append({
        "model_type": model_type,
        "score_type": "recall",
        "score": recall
    }, ignore_index=True)

    models_df = models_df.append({
        "model_type": model_type,
        "score_type": "f1 score",
        "score": f1
    }, ignore_index=True)

# 6: Display results graphically
fig, axes = plt.subplots(ncols=1, nrows=2)
fig.subplots_adjust(hspace=1)
axes[0].set_title("Recall and F1 Scores for each Model")
axes[1].set_title("Feature Importance")

# Adjust plot size
curr_size = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = (curr_size[0] * 3, curr_size[1] * 2)

# Recall and F1 Scores per model
ax1 = sns.barplot(x="model_type", y="score", hue="score_type", data=models_df, ax=axes[0])
# Feature importance metrics
ax2 = sns.barplot(x=X_train.columns, y=models[-1].feature_importances_, ax=axes[1])

plt.show()
