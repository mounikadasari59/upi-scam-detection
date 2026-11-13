import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
# Use the CSV filename you are training with.
# This file should have columns:
# message, language, label, scam_type
df = pd.read_csv("scam_messages.csv")

required_columns = {"message", "label", "scam_type"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        "Check that your CSV has a header row with "
        "message, language, label, scam_type."
    )

# --------------------------------------------------
# Clean dataset
# --------------------------------------------------
df["message"] = df["message"].astype("string").str.strip()
df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["message", "label"])
print("Duplicate messages:", df["message"].duplicated().sum())
print("Conflicting labels:", df.groupby("message")["label"].nunique().gt(1).sum())
df = df[df["message"] != ""]
df = df.drop_duplicates(subset=["message", "label"])
print("Messages after removing duplicates:", len(df))
df["label"] = df["label"].astype(int)

# Keep only valid binary labels
df = df[df["label"].isin([0, 1])].copy()

# Show dataset summary
print(f"Total messages: {len(df)}")
print(f"Legitimate messages: {(df['label'] == 0).sum()}")
print(f"Scam messages: {(df['label'] == 1).sum()}")

# --------------------------------------------------
# Show scam categories and sample messages
# --------------------------------------------------
scam_df = df[df["label"] == 1].copy()

print("\nScam categories in the dataset:")
print(scam_df["scam_type"].value_counts().to_string())

print("\nSample scam messages by category:")
for scam_type, group in scam_df.groupby("scam_type", dropna=False):
    category_name = str(scam_type)
    print(f"\n--- {category_name} ({len(group)} messages) ---")

    for message in group["message"].head(10):
        print(f"- {message}")

# --------------------------------------------------
# Split dataset into training and testing
# -------------------------------------------------
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df["message"], df["label"], groups=df["message"])
)

X_train = df["message"].iloc[train_idx]
X_test = df["message"].iloc[test_idx]
y_train = df["label"].iloc[train_idx]
y_test = df["label"].iloc[test_idx]

# --------------------------------------------------
# Create machine learning model
# --------------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# --------------------------------------------------
# Train model
# --------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate model
# --------------------------------------------------
predictions = model.predict(X_test)

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    predictions,
    labels=[0, 1],
    zero_division=0
)

accuracy = accuracy_score(y_test, predictions)

print("\nClassification Report (Out of 10)")
print("=" * 65)

print(
    f"{'Class':<15}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-score':<12}"
    f"{'Support':<10}"
)

classes = ["Legitimate", "Scam"]

for i, class_name in enumerate(classes):
    print(
        f"{class_name:<15}"
        f"{precision[i] * 10:<12.2f}"
        f"{recall[i] * 10:<12.2f}"
        f"{f1[i] * 10:<12.2f}"
        f"{support[i]:<10}"
    )

print("=" * 65)
print(f"Accuracy: {accuracy * 100:.2f}%")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
joblib.dump(model, "scam_model.pkl")
print("\nModel saved successfully as scam_model.pkl")