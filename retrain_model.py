"""
prompts_v2.csv 로 TF-IDF + Logistic Regression 모델을 재학습하고
models/ML Models/ 에 .pkl 파일을 저장합니다.

실행:
  .venv/bin/python retrain_model.py
"""

import string
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# ─── NLTK 리소스 ───────────────────────────────────────
for resource in ("punkt", "stopwords", "wordnet", "punkt_tab"):
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


def main():
    data_path = "data/raw/prompts_v2.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ {data_path} 파일이 없습니다. 먼저 generate_data.py를 실행하세요.")
        return

    print(f"데이터 로드: {len(df)}개 프롬프트, {df['cluster'].nunique()}개 클러스터, {df['sub_class'].nunique()}개 서브클래스")

    # ─── 전처리 ───────────────────────────────────────
    print("전처리 중...")
    df["processed"] = df["prompt"].apply(preprocess)

    # ─── 인코딩 ───────────────────────────────────────
    le_cluster   = LabelEncoder()
    le_sub_class = LabelEncoder()
    df["cluster_enc"]   = le_cluster.fit_transform(df["cluster"])
    df["sub_class_enc"] = le_sub_class.fit_transform(df["sub_class"])

    # ─── TF-IDF ───────────────────────────────────────
    print("TF-IDF 벡터화 중...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["processed"])

    # ─── 클러스터 모델 학습 ────────────────────────────
    print("클러스터 모델 학습 중...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df["cluster_enc"], test_size=0.2, stratify=df["cluster_enc"], random_state=42
    )
    cluster_model = LogisticRegression(max_iter=1000, C=1.0)
    cluster_model.fit(X_tr, y_tr)
    y_pred = cluster_model.predict(X_te)
    print(f"  클러스터 Accuracy: {accuracy_score(y_te, y_pred):.3f}  F1: {f1_score(y_te, y_pred, average='weighted'):.3f}")

    # ─── 서브클래스 모델 학습 ──────────────────────────
    print("서브클래스 모델 학습 중...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, df["sub_class_enc"], test_size=0.2, stratify=df["sub_class_enc"], random_state=42
    )
    subclass_model = LogisticRegression(max_iter=1000, C=1.0)
    subclass_model.fit(X_tr, y_tr)
    y_pred = subclass_model.predict(X_te)
    print(f"  서브클래스 Accuracy: {accuracy_score(y_te, y_pred):.3f}  F1: {f1_score(y_te, y_pred, average='weighted'):.3f}")

    # ─── 저장 ─────────────────────────────────────────
    print("모델 저장 중...")
    base = "models/ML Models"
    joblib.dump(cluster_model,  f"{base}/cluster_LR_model.pkl")
    joblib.dump(subclass_model, f"{base}/subclass_LR_model.pkl")
    joblib.dump(vectorizer,     f"{base}/tfidf_vectorizer.pkl")
    joblib.dump(le_cluster,     f"{base}/cluster_label_encoder.pkl")
    joblib.dump(le_sub_class,   f"{base}/subclass_label_encoder.pkl")
    print("✅ 완료! models/ML Models/ 에 저장됐습니다.")


if __name__ == "__main__":
    main()
