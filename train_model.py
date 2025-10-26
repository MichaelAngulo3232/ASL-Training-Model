import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading dataset...")
data = pd.read_csv("asl_data.csv")

data = data[~data["label"].isin(["nothing", "space", "del"])]

print("Class counts after filtering:")
print(data["label"].value_counts())

# Split features (X) and labels (y)
X = data.drop(columns=["label"])
y = data["label"]

# Train/test split so we can measure accuracy honestly
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,      # keeps label balance in train/test
    random_state=42
)

print("Training RandomForest...")
clf = RandomForestClassifier(
    n_estimators=200,   # number of trees
    random_state=42,
    n_jobs=-1          # use all CPU cores to go faster
)
clf.fit(X_train, y_train)

print("Evaluating...")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print()
print(classification_report(y_test, y_pred))

print("Saving model to asl_model.joblib ...")
joblib.dump(clf, "asl_model.joblib")
print("✅ Done. Model saved as asl_model.joblib")