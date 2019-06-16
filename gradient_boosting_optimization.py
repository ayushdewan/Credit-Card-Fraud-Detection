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

# 5: Hyperparameter search
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, log_loss

# 5a: learning rate search
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
lr_df = pd.DataFrame(columns=["learning_rate", "score", "type"])
loss_df = pd.DataFrame(columns=["learning_rate", "score", "type"])

for lr in learning_rates:
    print("Learning Rate:", lr)

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, max_depth=1, random_state=0)
    clf.fit(X_train_res, y_train_res)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    y_train_score = clf.decision_function(X_train)
    y_test_score = clf.decision_function(X_test)

    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)

    train_loss = log_loss(y_train, y_train_score)
    test_loss = log_loss(y_test, y_test_score)

    print("Train recall:", train_recall)
    print("Test recall:", test_recall)
    print("Train loss:", train_loss)
    print("Test loss:", test_loss, "\n")

    lr_df = lr_df.append({
        "learning_rate": lr,
        "score": train_recall,
        "type": "train_recall",
    }, ignore_index=True)

    lr_df = lr_df.append({
        "learning_rate": lr,
        "score": test_recall,
        "type": "test_recall",
    }, ignore_index=True)

    loss_df = loss_df.append({
        "learning_rate": lr,
        "score": train_loss,
        "type": "train_loss",
    }, ignore_index=True)

    loss_df = loss_df.append({
        "learning_rate": lr,
        "score": test_loss,
        "type": "test_loss",
    }, ignore_index=True)
print(lr_df.head())
sns.lineplot(x="learning_rate", y="score", hue="type", data=lr_df)
plt.show()

sns.lineplot(x="learning_rate", y="score", hue="type", data=loss_df)
plt.show()


# 5b: n_estimators search
estimators = [1, 2, 4, 8, 16, 32, 64, 100]
ne_df = pd.DataFrame(columns=["n_estimators", "score", "type"])
ne_loss_df = pd.DataFrame(columns=["n_estimators", "score", "type"])

for ne in estimators:
    print("n_estimators:", ne)

    clf = GradientBoostingClassifier(n_estimators=ne, learning_rate=0.5, max_depth=1, random_state=0)
    clf.fit(X_train_res, y_train_res)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    y_train_score = clf.decision_function(X_train)
    y_test_score = clf.decision_function(X_test)

    train_recall = recall_score(y_train, y_train_pred)
    test_recall = recall_score(y_test, y_test_pred)

    train_loss = log_loss(y_train, y_train_score)
    test_loss = log_loss(y_test, y_test_score)

    print("Train recall:", train_recall)
    print("Test recall:", test_recall)
    print("Train loss:", train_loss)
    print("Test loss:", test_loss, "\n")

    ne_df = ne_df.append({
        "n_estimators": ne,
        "score": train_recall,
        "type": "train_recall",
    }, ignore_index=True)

    ne_df = ne_df.append({
        "n_estimators": ne,
        "score": test_recall,
        "type": "test_recall",
    }, ignore_index=True)

    ne_loss_df = ne_loss_df.append({
        "n_estimators": ne,
        "score": train_loss,
        "type": "train_loss",
    }, ignore_index=True)

    ne_loss_df = ne_loss_df.append({
        "n_estimators": ne,
        "score": test_loss,
        "type": "test_loss",
    }, ignore_index=True)
    
sns.lineplot(x="n_estimators", y="score", hue="type", data=ne_df)
plt.show()

sns.lineplot(x="n_estimators", y="score", hue="type", data=ne_loss_df)
plt.show()

# Best parameters: learnign_rate = 0.5, n_estimators = s100