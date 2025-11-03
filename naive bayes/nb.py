import numpy as np
import pandas as pd


def cond_prob(df, y, feature, value, c):
    """Compute P(feature=value | class=c)."""
    df_c = df[df[y] == c] # only rows where class = c
    denominator = len(df_c)
    num = int((df_c[feature] == value).sum()) # within the rows where class = c, how many have feature = value?
    return num / denominator


def posteriors_for_sample(df, X, y, values):
    """Compute P(class=c | sample=x)."""
    sample = {f: str(v) for f, v in zip(X, values)}  # Pairs up each feature name with its corresponding sample value.
    classes = sorted(df[y].unique().tolist())
    priors = {c: (df[y] == c).mean() for c in classes}

    numerators = {}
    for c in classes:
        p = priors[c]  # p = P(c), prior probability of class c
        for f in X:
            p *= cond_prob(df, y, f, sample[f], c)  # p = P(c) × ΠP(x_i=v_i|c), numerator in Bayes' law
        numerators[c] = p

    Z = sum(numerators.values())  # P(c|x) = P(c) × ΠP(x_i=v_i|c) / P(x)
    return {c: numerators[c] / Z for c in classes}


def predict(posteriors):
    """Return the class with the highest posterior probability."""
    return max(posteriors.items(), key=lambda kv: kv[1])[0] # the chosen class is the one with the highest posterior


# ---- main script ----
CSV_PATH = "naive_bayes_example.csv"
df = pd.read_csv(CSV_PATH).astype(str)

y = df.columns[-1]
X = list(df.columns[:-1])

NEW_SAMPLE = ["Young", "Paint", "cold"]

posts = posteriors_for_sample(df, X, y, NEW_SAMPLE)
for c, prob in posts.items():
    print(f"P({c}|x) = {prob:.3f}")
print(f"\nPredicted class: {predict(posts)}")
