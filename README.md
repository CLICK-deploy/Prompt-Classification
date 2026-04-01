## Intelligent Prompt Classification for AI Task Routing

> **CLiCK 프로젝트 적용 버전**
> 기존 4개 클러스터 / 10개 서브클래스 기반의 TF-IDF 모델에서, CLiCK 프로젝트 요구사항에 맞춰 **40개 카테고리 / 다국어(한국어·영어) 지원 Sentence-Transformer 기반 분류 모델**로 전면 교체됐습니다.

**Project Description**

The objective is to classify user prompts into predefined categories and sub-categories. This classification is crucial for routing user requests to the most appropriate AI platform or tool, enhancing the efficiency and accuracy of AI-powered applications.

**Dataset**

- `prompts_v1.csv`: 원본 합성 데이터셋 (~316개 프롬프트, 4개 클러스터, 10개 서브클래스, 영어 전용)
- `prompts_v2.csv`: CLiCK 적용을 위해 OpenAI API로 생성된 확장 데이터셋 (40개 카테고리 × 20개 = 800개+, 한국어·영어 혼합) — `generate_data.py`로 생성

**Categories (CLiCK 적용 버전, 40개)**

| 도메인 | 카테고리 |
|---|---|
| 코딩/기술 | 코드 작성, 코드 디버깅/리뷰, 인프라/DevOps |
| 글쓰기/문서 | 창작/소설, 비즈니스 문서/이메일, 블로그/SNS |
| 분석/리서치 | 데이터 분석, 비교/평가, 사실 확인/팩트체크 |
| 학습/교육 | 개념 설명, 언어 학습, 시험/퀴즈 |
| 비즈니스 | 마케팅/광고, 기획/전략, 법률/계약 |
| 일상/생활 | 요리/레시피, 여행/장소, 건강/의학 |
| 창의/엔터테인먼트 | 아이디어 브레인스토밍, 롤플레이/페르소나, 음악/영화/게임 |
| 소통/심리 | 감정 상담, 자기계발, 관계/커뮤니케이션 |
| 과학/공학 | 수학/통계, 물리/화학/생물 |
| 디자인/시각 | UI/UX, 이미지 프롬프트 생성 |
| 금융/경제 | 재무/회계, 투자/주식 |
| 법/사회 | 법률/계약, 정치/사회 이슈 |
| 환경/자연 | 환경/기후, 동물/식물 |
| 스포츠/피트니스 | 운동/트레이닝, 스포츠 정보 |
| 종교/철학 | 철학/윤리, 종교/영성 |
| 기타 | 보안/해킹, 기타 |

**Methodologies**

Three distinct approaches were implemented and evaluated for prompt classification:

1. **Fine-tuned BERT for Hierarchical Classification**

   * **Approach:** A pre-trained BERT model that was fine-tuned using the Hugging Face `transformers` library. Two separate AutoModelForSequenceClassification models were trained: one for cluster classification and another for sub-classification within the predicted cluster.
   * **Strengths:** BERT's ability to capture contextual information in language makes it well-suited for this task. With sufficient data, fine-tuned BERT models can achieve state-of-the-art results in text classification.
   * **Limitations:** Computationally expensive, especially with larger datasets and complex models. Prone to overfitting on smaller datasets.

2. **Traditional Machine Learning**

   * **Approach:**  Classic machine learning algorithms (Logistic Regression, SVM, Naive Bayes) were trained on TF-IDF (Term Frequency-Inverse Document Frequency) vectors extracted from the prompts. Preprocessing included punctuation removal, stop word removal, and lemmatization.
   * **Strengths:**  Relatively simple to implement and computationally less demanding than deep learning approaches. Logistic Regression provided the best results in this case.
   * **Limitations:**  May not capture complex language nuances as effectively as deep learning models. Requires significant feature engineering and a large amount of data for optimal performance.

3. **Prompt Engineering with Large Language Models (LLMs)**

   * **Approach:**  Leveraged the power of LLMs (using OpenAI's API) with carefully crafted prompt templates for each cluster and sub-category. A two-stage prompting approach was used: first to identify the cluster and then to predict the sub-class using a cluster-specific prompt.
   * **Strengths:**  Can achieve impressive results with less training data than traditional methods. Highly adaptable and can be easily extended with new categories or tasks.
   * **Limitations:**  Relies on the quality of prompt engineering and the inherent uncertainty of LLM outputs. Can be more challenging to systematically evaluate and debug compared to traditional models.

4. **Multilingual Sentence-Transformer (CLiCK 적용 버전)**

   * **Approach:** `paraphrase-multilingual-MiniLM-L12-v2` 모델로 프롬프트와 카테고리 설명문을 각각 임베딩한 뒤, 코사인 유사도를 이용해 가장 가까운 카테고리로 분류합니다. 별도 학습 없이 zero-shot으로 동작하며 한국어/영어를 모두 지원합니다.
   * **Strengths:** 재학습 없이 카테고리 추가·수정 가능. 한국어 프롬프트 처리 가능. TF-IDF vocabulary 外 단어에도 강인함.
   * **Limitations:** 카테고리 설명문의 품질에 따라 정확도가 달라짐. 임베딩 모델 로드 시 초기 지연 발생.

**Evaluation and Ranking (원본 데이터 기준)**

The performance of each approach was evaluated using accuracy and F1-score metrics:

| Approach                               | Accuracy | F1-Score |
|----------------------------------------|----------|----------|
| 1. Fine-tuned BERT                     | 99%      | 98%      |
| 2. Traditional ML (Logistic Regression) | 96.8%    | 96.7%    |
| 3. LLMs and Prompt Engineering        | 84%      | 86%      |
| 4. Sentence-Transformer (CLiCK)       | zero-shot | -       |

**Quick Start (CLiCK 적용 버전)**

```bash
# 패키지 설치
pip install sentence-transformers scikit-learn pandas joblib

# 분류기 실행
python _test_predict.py
```

데이터 재생성 및 모델 재학습이 필요한 경우:

```bash
# 1. 학습 데이터 생성 (OpenAI API 키 필요)
export OPENAI_API_KEY=sk-...
python generate_data.py

# 2. 모델 재학습
python retrain_model.py
```

**Conclusion**

This project demonstrates the effectiveness of different approaches for prompt classification. The choice of the best approach depends on factors such as dataset size, computational resources, and desired accuracy levels. For the CLiCK project, the Sentence-Transformer based approach was adopted to support multilingual prompts (Korean/English) and an expanded 40-category taxonomy without requiring model retraining.
