"""
40개 카테고리에 대한 학습 데이터를 OpenAI API로 생성하고
data/raw/prompts_v2.csv 로 저장합니다.

실행 전 OPENAI_API_KEY 환경변수를 설정하세요:
  export OPENAI_API_KEY=sk-...

실행:
  .venv/bin/python generate_data.py
"""

import os
import json
import time
import pandas as pd
from openai import OpenAI

# ─────────────────────────────────────────────
# 40개 카테고리 정의 (domain, name, name_en)
# ─────────────────────────────────────────────
CATEGORIES = [
    ("코딩/기술",           "코드 작성",                "code_writing"),
    ("코딩/기술",           "코드 디버깅/리뷰",          "code_debug_review"),
    ("코딩/기술",           "인프라/DevOps",            "infra_devops"),
    ("글쓰기/문서",          "창작/소설",                "creative_writing"),
    ("글쓰기/문서",          "비즈니스 문서/이메일",       "business_document"),
    ("글쓰기/문서",          "블로그/SNS",               "blog_sns"),
    ("분석/리서치",          "데이터 분석",               "data_analysis"),
    ("분석/리서치",          "비교/평가",                 "comparison"),
    ("분석/리서치",          "사실 확인/팩트체크",          "fact_check"),
    ("학습/교육",            "개념 설명",                 "concept_explanation"),
    ("학습/교육",            "언어 학습",                 "language_learning"),
    ("학습/교육",            "시험/퀴즈",                 "exam_quiz"),
    ("비즈니스",             "마케팅/광고",               "marketing"),
    ("비즈니스",             "기획/전략",                 "planning_strategy"),
    ("비즈니스",             "법률/계약",                 "legal"),
    ("일상/생활",            "요리/레시피",               "cooking"),
    ("일상/생활",            "여행/장소",                 "travel"),
    ("일상/생활",            "건강/의학",                 "health_medical"),
    ("창의/엔터테인먼트",     "아이디어 브레인스토밍",       "brainstorming"),
    ("창의/엔터테인먼트",     "롤플레이/페르소나",           "roleplay"),
    ("창의/엔터테인먼트",     "음악/영화/게임",             "entertainment"),
    ("소통/심리",            "감정 상담",                 "emotional_counseling"),
    ("소통/심리",            "자기계발",                  "self_improvement"),
    ("소통/심리",            "관계/커뮤니케이션",           "relationship"),
    ("과학/공학",            "수학/통계",                 "math_statistics"),
    ("과학/공학",            "물리/화학/생물",             "science"),
    ("디자인/시각",          "UI/UX",                    "ui_ux"),
    ("디자인/시각",          "이미지 프롬프트 생성",        "image_prompt"),
    ("금융/경제",            "재무/회계",                 "finance_accounting"),
    ("금융/경제",            "투자/주식",                 "investment"),
    ("법/사회",              "법률/계약",                 "law_contract"),
    ("법/사회",              "정치/사회 이슈",             "politics_society"),
    ("환경/자연",            "환경/기후",                 "environment"),
    ("환경/자연",            "동물/식물",                 "animals_plants"),
    ("스포츠/피트니스",       "운동/트레이닝",              "exercise_training"),
    ("스포츠/피트니스",       "스포츠 정보",               "sports_info"),
    ("종교/철학",            "철학/윤리",                 "philosophy_ethics"),
    ("종교/철학",            "종교/영성",                 "religion_spirituality"),
    ("기타",                "보안/해킹",                 "security_hacking"),
    ("기타",                "기타",                      "etc"),
]

PROMPTS_PER_CATEGORY = 20  # 카테고리당 생성할 프롬프트 수

SYSTEM_MSG = "You are a dataset generator. Generate realistic user prompts for AI assistants."

def generate_prompts(client: OpenAI, domain: str, name: str, n: int) -> list[str]:
    user_msg = f"""Generate {n} realistic and diverse user prompts that belong to the category:
Domain: {domain}
Category: {name}

Requirements:
- Each prompt should be a realistic question or request a user might send to an AI assistant
- Mix English and Korean prompts (roughly 50/50)
- Make them varied in length and style
- Do NOT number them
- Return ONLY a JSON array of strings, nothing else

Example format: ["prompt 1", "prompt 2", ...]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    # JSON object 안에 배열이 들어있는 경우 처리
    parsed = json.loads(content)
    if isinstance(parsed, list):
        return parsed
    # {"prompts": [...]} 형태인 경우
    for v in parsed.values():
        if isinstance(v, list):
            return v
    return []


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export OPENAI_API_KEY=sk-... 후 다시 실행하세요.")
        return

    client = OpenAI(api_key=api_key)
    rows = []

    total = len(CATEGORIES)
    for i, (domain, name, name_en) in enumerate(CATEGORIES, 1):
        print(f"[{i:02d}/{total}] {domain} > {name} 생성 중...", end=" ", flush=True)
        try:
            prompts = generate_prompts(client, domain, name, PROMPTS_PER_CATEGORY)
            for p in prompts:
                rows.append({"prompt": p, "cluster": domain, "sub_class": name})
            print(f"✅ {len(prompts)}개")
        except Exception as e:
            print(f"❌ 실패: {e}")

        # API rate limit 방지
        if i % 10 == 0:
            time.sleep(2)

    df = pd.DataFrame(rows)
    out_path = "data/raw/prompts_v2.csv"
    df.to_csv(out_path, index=False)
    print(f"\n완료: {len(df)}개 프롬프트 → {out_path}")
    print(df["cluster"].value_counts())


if __name__ == "__main__":
    main()
