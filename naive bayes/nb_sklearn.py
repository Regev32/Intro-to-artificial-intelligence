import pandas as pd
from sklearn.naive_bayes import CategoricalNB

# Helper function to encode new samples
def encode_sample(sample_dict, cats_dict):
    """
    Turn a dictionary like {"Age": "Young", "Hobby": "Paint", "Weather": "cold"}
    into a list of integer codes that match the training encoding.
    """
    encoded = []
    for col in FEATURES:
        value = sample_dict[col]
        if value not in cats_dict[col]:
            raise ValueError(
                f"Value '{value}' not seen in training for column '{col}'. "
                f"Known categories: {cats_dict[col]}"
            )
        encoded.append(cats_dict[col].index(value))
    return encoded

# 1) Load the dataset
CSV_PATH = "../naive bayes/naive_bayes_example.csv"  # Path to the CSV file
df = pd.read_csv(CSV_PATH)

# Define the input features and the target label
FEATURES = ["Age", "Hobby", "Weather"]
TARGET = "Buy?"

X_text = df[FEATURES].copy()
y = df[TARGET].copy()

# 2) Encode text features as integers
cats = {}
X_enc = pd.DataFrame()

for col in FEATURES:
    cat_series = pd.Categorical(X_text[col])
    X_enc[col] = cat_series.codes  # convert to numeric codes
    cats[col] = list(cat_series.categories)  # save the category names

# 3) Train the Naive Bayes model
nb = CategoricalNB(alpha=0)
nb.fit(X_enc.values, y.values)

# 4) Predict a new sample
NEW_SAMPLE = {"Age": "Young", "Hobby": "Paint", "Weather": "cold"}
x_new_enc = [encode_sample(NEW_SAMPLE, cats)]

# Compute class probabilities and predicted label
proba = nb.predict_proba(x_new_enc)[0]
pred = nb.predict(x_new_enc)[0]

# 5) Print the results
print("Class probabilities:")
for cls, p in zip(nb.classes_, proba):
    print(f"  P({cls}|x) = {p:.3f}")

print(f"\nPredicted class: {pred}")
