import joblib
import pandas as pd

df = pd.read_csv("data/raw/prompts_v1.csv")
cluster_LR = joblib.load("models/ML Models/cluster_LR_model.pkl")
cluster_encoder = joblib.load("models/ML Models/cluster_label_encoder.pkl")
subclass_LR = joblib.load("models/ML Models/subclass_LR_model.pkl")
subclass_encoder = joblib.load("models/ML Models/subclass_label_encoder.pkl")
vectorizer = joblib.load("models/ML Models/tfidf_vectorizer.pkl")

print(f"학습 데이터 샘플:\n{df['prompt'].head(3).tolist()}\n")
print(f"클러스터 종류: {list(cluster_encoder.classes_)}\n")
print(f"subclass 종류: {list(subclass_encoder.classes_)}\n")

prompts = [
    "recently, id like to buy a new labtop. can you recommend me something?",
    "well, how about my launch?",
    "Stock market prediction for tomorrow",
    "How to use OpenAI API?",
]

for p in prompts:
    X = vectorizer.transform([p])
    cp = cluster_LR.predict_proba(X)
    sp = subclass_LR.predict_proba(X)
    ci = cp.argmax(axis=1)
    si = sp.argmax(axis=1)
    print(f'입력: "{p[:50]}"')
    print(f'  cluster  → {cluster_encoder.inverse_transform(ci)[0]} (confidence: {cp[0][ci][0]:.3f})')
    print(f'  subclass → {subclass_encoder.inverse_transform(si)[0]} (confidence: {sp[0][si][0]:.3f})')
    print()
