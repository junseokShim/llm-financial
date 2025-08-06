import re
import requests
from bs4 import BeautifulSoup

# 객관식 여부 판단 함수
def is_multiple_choice(question_text):
    """
    객관식 여부를 판단: 2개 이상의 숫자 선택지가 줄 단위로 존재할 경우 객관식으로 간주
    """
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2


# 질문과 선택지 분리 함수
def extract_question_and_choices(full_text):
    """
    전체 질문 문자열에서 질문 본문과 선택지 리스트를 분리
    """
    lines = full_text.strip().split("\n")
    q_lines = []
    options = []

    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    
    question = " ".join(q_lines)
    return question, options


def make_prompt_auto(text):
# 프롬프트 생성기
    additional_info = web_search(text)
    
    if is_multiple_choice(text):
                
        question, options = extract_question_and_choices(text)
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                "제공된 추가 정보와 질문을 고려하여 아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
                f"추가 정보: {additional_info}\n\n"
                f"질문: {question}\n"
                "선택지:\n"
                f"{chr(10).join(options)}\n\n"
                "답변:"
                )
    else:
        prompt = (
                "당신은 금융보안 전문가입니다.\n"
                " 제공된 추가 정보를 참고하여 아래 주관식 질문에 대해 정확하고 간략한 설명을 작성하세요.\n\n"
                f"추가 정보: {additional_info}\n\n"
                f"질문: {text}\n\n"
                "답변:"
                )   
    return prompt


# 인터넷 검색을 통한 추가 정보 수집 함수
def web_search(query):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://www.google.com/search?q={query}"
        response = requests.get(search_url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        snippets = soup.select('div.BNeawe.s3v9rd.AP7Wnd')
        results = ' '.join(snippet.get_text() for snippet in snippets[:3])
        return results
    except requests.exceptions.RequestException:
        return ""