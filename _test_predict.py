import re

# ─────────────────────────────────────────────
# 카테고리 정의 (category_classifier.cpp 기반, 40개)
# TF-IDF는 영어 기반이므로 영어 키워드 위주로 작성합니다.
# ─────────────────────────────────────────────
CLUSTERS = {
    # ── 코딩/기술 ──────────────────────────────
    "코딩/기술 > 코드 작성":           "code function class implement write program script coding",
    "코딩/기술 > 코드 디버깅/리뷰":     "bug error debug fix review not working issue exception trace",
    "코딩/기술 > 인프라/DevOps":        "server deploy docker kubernetes CI CD cloud AWS GCP Azure linux shell nginx pipeline",

    # ── 글쓰기/문서 ────────────────────────────
    "글쓰기/문서 > 창작/소설":           "novel story fiction character fantasy poem script creative write",
    "글쓰기/문서 > 비즈니스 문서/이메일": "email proposal report document template memo business write letter",
    "글쓰기/문서 > 블로그/SNS":          "blog instagram twitter post content caption hashtag youtube social media",

    # ── 분석/리서치 ────────────────────────────
    "분석/리서치 > 데이터 분석":         "data analysis chart graph statistics visualization excel CSV pandas numpy dataset",
    "분석/리서치 > 비교/평가":           "compare difference pros cons better evaluate ranking review versus",
    "분석/리서치 > 사실 확인/팩트체크":   "fact true false verify source accurate check confirm",

    # ── 학습/교육 ──────────────────────────────
    "학습/교육 > 개념 설명":             "explain what is how does why concept definition meaning understand",
    "학습/교육 > 언어 학습":             "english japanese chinese grammar vocabulary pronunciation language translate",
    "학습/교육 > 시험/퀴즈":             "quiz exam test problem question practice answer solve",

    # ── 비즈니스 ───────────────────────────────
    "비즈니스 > 마케팅/광고":            "marketing advertising branding campaign slogan copywriting sales customer",
    "비즈니스 > 기획/전략":              "planning strategy roadmap goal OKR KPI startup business model",
    "비즈니스 > 법률/계약":              "law contract legal regulation copyright patent compliance policy",

    # ── 일상/생활 ──────────────────────────────
    "일상/생활 > 요리/레시피":           "recipe cooking food ingredient bake cook dish meal",
    "일상/생활 > 여행/장소":             "travel trip hotel flight restaurant tourist itinerary destination",
    "일상/생활 > 건강/의학":             "health symptom doctor medicine treatment diet nutrition fitness",

    # ── 창의/엔터테인먼트 ──────────────────────
    "창의/엔터테인먼트 > 아이디어 브레인스토밍": "idea brainstorm creative suggest innovative think generate new",
    "창의/엔터테인먼트 > 롤플레이/페르소나":     "roleplay persona act as pretend character simulate",
    "창의/엔터테인먼트 > 음악/영화/게임":        "music song movie drama game recommend lyrics anime entertainment",

    # ── 소통/심리 ──────────────────────────────
    "소통/심리 > 감정 상담":             "depressed anxious stressed lonely sad comfort feeling mental health",
    "소통/심리 > 자기계발":              "self improvement habit motivation productivity growth routine mindset",
    "소통/심리 > 관계/커뮤니케이션":      "relationship friend love family conflict communication persuade",

    # ── 과학/공학 ──────────────────────────────
    "과학/공학 > 수학/통계":             "math calculate formula equation probability statistics matrix vector",
    "과학/공학 > 물리/화학/생물":         "physics chemistry biology molecule DNA cell energy force reaction",

    # ── 디자인/시각 ────────────────────────────
    "디자인/시각 > UI/UX":              "UI UX design interface layout wireframe figma color font usability",
    "디자인/시각 > 이미지 프롬프트 생성": "image picture prompt dall-e midjourney stable diffusion generate AI art",

    # ── 금융/경제 ──────────────────────────────
    "금융/경제 > 재무/회계":             "finance accounting tax budget revenue cost balance sheet profit",
    "금융/경제 > 투자/주식":             "invest stock crypto bitcoin ETF fund real estate portfolio dividend",

    # ── 법/사회 ────────────────────────────────
    "법/사회 > 법률/계약":               "lawsuit verdict civil criminal labor law rights contract legal",
    "법/사회 > 정치/사회 이슈":          "politics society election government policy issue human rights welfare",

    # ── 환경/자연 ──────────────────────────────
    "환경/자연 > 환경/기후":             "environment climate carbon warming renewable pollution sustainable energy",
    "환경/자연 > 동물/식물":             "animal plant pet dog cat flower tree ecosystem wildlife nature",

    # ── 스포츠/피트니스 ────────────────────────
    "스포츠/피트니스 > 운동/트레이닝":   "exercise workout training muscle fitness yoga cardio strength",
    "스포츠/피트니스 > 스포츠 정보":     "soccer baseball basketball tennis golf sports match player team league",

    # ── 종교/철학 ──────────────────────────────
    "종교/철학 > 철학/윤리":             "philosophy ethics moral existence consciousness free will truth value",
    "종교/철학 > 종교/영성":             "religion god christianity buddhism islam spirituality meditation prayer",

    # ── 기타 ───────────────────────────────────
    "기타 > 보안/해킹":                 "security hacking vulnerability encryption firewall malware phishing CTF",
    "기타 > 기타":                      "general question miscellaneous other advice recommendation",
}

# ─────────────────────────────────────────────
# 임베딩 모델 로드 (최초 1회 다운로드 후 캐시 사용)
# paraphrase-multilingual-MiniLM-L12-v2: 한국어/영어 모두 지원
# ─────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("임베딩 모델 로드 중...", end=" ", flush=True)
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("완료")

cluster_names = list(CLUSTERS.keys())
cluster_vectors = embedder.encode(list(CLUSTERS.values()), convert_to_numpy=True)

def classify_prompt(prompt):
    prompt_vec = embedder.encode([prompt], convert_to_numpy=True)
    scores = cosine_similarity(prompt_vec, cluster_vectors)[0]
    best_idx = int(np.argmax(scores))
    return cluster_names[best_idx], float(scores[best_idx])

print("=== Prompt Classifier ===")
print(f"카테고리 {len(CLUSTERS)}개 로드 완료")
print("종료하려면 'q' 또는 'quit' 입력\n")

while True:
    prompt = input("프롬프트 입력: ").strip()
    if prompt.lower() in ("q", "quit", "exit"):
        print("종료합니다.")
        break
    if not prompt:
        continue
    cluster, score = classify_prompt(prompt)
    print(f"결과: {cluster}  (score: {score:.3f})\n")
