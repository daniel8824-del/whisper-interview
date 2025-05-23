# ===== 1단계: 초기 설정 및 환경 구성 =====
import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import logging
import sys
from pydub import AudioSegment
import math

# 환경 변수 로드 및 OpenAI API 키 설정
load_dotenv()

# OpenAI API 클라이언트 초기화
client = OpenAI()

# 로깅 설정 - 파일 및 콘솔 출력 구성
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interview_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===== 2단계: 음성 데이터 처리 =====
# ----- 2.1 오디오 파일 분할 -----
def split_audio(audio_file_path, max_size_mb=25):
    """
    대용량 오디오 파일을 Whisper API 제한(25MB)에 맞게 분할하는 함수
    
    Args:
        audio_file_path (str): 분할할 오디오 파일 경로
        max_size_mb (int): 각 청크의 최대 크기 (MB)
        
    Returns:
        list: 분할된 오디오 파일 경로 목록
    """
    try:
        # 파일 크기 확인 (바이트)
        file_size = os.path.getsize(audio_file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # 파일이 최대 크기보다 작으면 분할 불필요
        if file_size <= max_size_bytes:
            return [audio_file_path]
            
        logger.info(f"오디오 파일 분할 시작: {audio_file_path}")
        
        # 오디오 파일 로드
        audio = AudioSegment.from_file(audio_file_path)
        duration_ms = len(audio)
        
        # 청크 개수 계산
        num_chunks = math.ceil(file_size / max_size_bytes)
        chunk_duration = duration_ms // num_chunks
        
        # 청크 파일 저장
        chunk_files = []
        for i in range(num_chunks):
            start_ms = i * chunk_duration
            end_ms = min((i + 1) * chunk_duration, duration_ms)
            
            chunk = audio[start_ms:end_ms]
            chunk_path = f"{audio_file_path}_chunk_{i+1}.wav"
            chunk.export(chunk_path, format="wav")
            chunk_files.append(chunk_path)
            
        logger.info(f"오디오 파일이 {len(chunk_files)}개의 청크로 분할되었습니다.")
        return chunk_files
        
    except Exception as e:
        logger.error(f"오디오 파일 분할 중 오류 발생: {str(e)}", exc_info=True)
        return [audio_file_path]

# ----- 2.2 음성을 텍스트로 변환 -----
def get_transcript_only(audio_file_path):
    """
    Whisper API를 사용하여 음성 파일을 텍스트로 변환하는 함수
    분할된 오디오 파일들을 순차적으로 처리하고 결과를 통합
    
    Args:
        audio_file_path (str): 분석할 음성 파일 경로
        
    Returns:
        str: 추출된 텍스트
        None: 오류 발생 시
    """
    try:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_file_path}")
            
        logger.info(f"음성 파일 텍스트 추출 시작: {audio_file_path}")
        
        # 파일 분할
        chunk_files = split_audio(audio_file_path)
        full_transcript = ""
        
        # 각 청크에서 텍스트 추출
        for chunk_file in chunk_files:
            with open(chunk_file, "rb") as audio_file:
                chunk_transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko",
                    response_format="text"
                )
                full_transcript += chunk_transcript + " "
            
            # 임시 청크 파일 삭제
            if chunk_file != audio_file_path:
                os.remove(chunk_file)
        
        logger.info("텍스트 추출 완료")
        return full_transcript.strip()
        
    except Exception as e:
        logger.error(f"텍스트 추출 중 오류 발생: {str(e)}", exc_info=True)
        return None

# ===== 3단계: 텍스트 분석 =====
# ----- 3.1 텍스트 구조화 -----
def create_structured_analysis(transcript):
    """
    GPT-4를 사용하여 면접 답변을 구조화된 형태로 변환
    질문/답변 구분, 요약, 감정 상태, 의도 등을 분석
    
    Args:
        transcript (str): 변환할 텍스트
        
    Returns:
        dict: 구조화된 데이터
        str: 오류 발생 시 오류 메시지
    """
    try:
        # 기본 구조 생성
        structured_data = {
            "dataSet": {
                "question": {
                    "raw": {"text": ""},
                    "intent": []
                },
                "answer": {
                    "raw": {"text": transcript},
                    "summary": {"text": ""},
                    "emotion": {},
                    "intent": [{"category": "일반"}]
                },
                "info": {
                    "occupation": "일반"
                }
            }
        }

        # GPT를 사용하여 텍스트를 구조화
        structure_prompt = f"""
당신은 채용담당자입니다. 주어진 면접 답변 내용을 정리해주세요.
응답은 반드시 JSON 형식이어야 합니다.

답변 내용:
{transcript}

다음 형식으로 분류해주세요:
1. 질문과 답변 구분
2. 각 답변의 요약
3. 답변의 감정 상태 (positive/negative/neutral)
4. 주요 의도와 카테고리
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 채용담당자입니다. 주어진 면접 답변 내용을 JSON 형식으로 정리해주세요"},
                {"role": "user", "content": structure_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        structured_result = json.loads(response.choices[0].message.content)
        structured_data.update(structured_result)
        return structured_data
        
    except Exception as e:
        logger.error(f"구조화 중 오류 발생: {str(e)}", exc_info=True)
        return f"구조화 중 오류 발생: {str(e)}"

# ----- 3.2 상세 분석 -----
def analyze_interview_data(structured_data):
    """
    GPT-4를 사용하여 구조화된 면접 데이터를 상세 분석
    - 답변 분석: 논리성, 구체성, 전달력
    - 의도 분석: 관련성, 초점
    - 감정 분석: 톤, 단어 선택
    - 역량 평가: 전문성, 의사소통
    - 키워드 분석: 긍정 표현, 전문 용어, 주요 키워드
    - 답변 통계: 단어수, 문장수, 답변 시간
    
    Args:
        structured_data (dict): 구조화된 면접 데이터
        
    Returns:
        dict: 분석 결과
        None: 오류 발생 시
    """
    try:
        # 분석을 위한 프롬프트 생성
        analysis_prompt = f"""
당신은 면접관입니다. 다음 면접 답변을 분석하여 JSON 형식으로 응답해주세요:

답변 내용: {structured_data['dataSet']['answer']['raw']['text']}

응답은 반드시 다음의 JSON 형식을 따라야 합니다:
{{
    "answer_analysis": {{
        "logic": {{
            "score": "문장 구조, 일관성, 주장-근거 연결성 평가 (0-5점)",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "specificity": {{
            "score": "구체적 사례, 키워드 사용, 맥락 설명 수준 (0-5점)",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "delivery": {{
            "score": "문장 완성도, 핵심 단어 사용, 설명 명확성 (0-5점)",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "subtotal": "답변 분석 총점 (0-15점)"
    }},
    "intent_analysis": {{
        "relevance": {{
            "score": "질문-답변 매칭도, 키워드 일치도 (0-10점)",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "focus": {{
            "score": "핵심 포인트 강조, 불필요 내용 최소화 (0-5점)",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "subtotal": "의도 분석 총점 (0-15점)"
    }},
    "emotion_analysis": {{
        "tone": {{
            "score": "감정 톤 분석, 자신감 표현 (0-10점)",
            "sentiment": "긍정/부정/중립 비율",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "word_choice": {{
            "score": "감정 표현 단어, 강조/완화 표현 (0-5점)",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "subtotal": "감정 분석 총점 (0-15점)"
    }},
    "competency_evaluation": {{
        "expertise": {{
            "score": "전문 용어 활용도, 업무 이해도 (0-10점)",
            "keywords": ["발견된 전문 용어1", "발견된 전문 용어2"],
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "communication": {{
            "score": "설명 능력, 표현의 적절성 (0-5점)",
            "details": ["평가 근거1", "평가 근거2"]
        }},
        "subtotal": "역량 평가 총점 (0-15점)"
    }},
    "keyword_analysis": {{
        "positive_ratio": "긍정 표현 비율 (0-100%)",
        "professional_terms_count": "전문 용어 개수",
        "main_keywords": ["주요 키워드1", "주요 키워드2", "주요 키워드3", "주요 키워드4", "주요 키워드5"],
        "emotional_expressions": ["감정 표현 단어1", "감정 표현 단어2", "감정 표현 단어3"]
    }},
    "answer_statistics": {{
        "total_words": "총 단어수 (숫자)",
        "total_sentences": "총 문장수 (숫자)",
        "avg_sentence_length": "평균 문장 길이 (단어수)",
        "estimated_time": "예상 답변 시간 (초)",
        "detail_level": "답변 상세도 (높음/보통/낮음)"
    }},
    "total_score": {{
        "score": "총점 (0-60점)",
        "level": "수준 (탁월/우수/보통/미흡)",
        "summary": "종합 평가 의견",
        "key_strengths": ["반드시 3가지의 긍정적인 면을 서술해주세요. 가장 두드러진 장점부터 순서대로 작성하세요."],
        "improvements": ["반드시 3가지의 부정적인 면을 서술해주세요. 가장 시급한 개선이 필요한 부분부터 순서대로 작성하세요."]
    }}
}}

분석 지침:
1. keyword_analysis: 답변에서 실제로 사용된 긍정적 표현의 비율을 계산하고, 전문 용어를 정확히 찾아내세요.
2. main_keywords: 답변에서 가장 중요하고 의미있는 5개의 키워드를 추출하세요.
3. answer_statistics: 실제 단어수와 문장수를 정확히 계산하고, 예상 답변 시간은 단어수 × 0.5초로 계산하세요.
4. detail_level: 답변의 구체성과 설명의 깊이를 바탕으로 평가하세요.
5. key_strengths와 improvements는 각각 정확히 3가지씩 작성해야 합니다.
"""

        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": "당신은 면접관입니다. 면접 답변을 상세히 분석하여 JSON 형식으로 응답해주세요."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        analysis_result = json.loads(response.choices[0].message.content)
        return analysis_result
        
    except Exception as e:
        logger.error(f"답변 분석 중 오류 발생: {str(e)}", exc_info=True)
        return None

# ===== 4단계: 결과 생성 =====
# ----- 4.1 HTML 리포트 생성 -----
def save_as_html_report(result, output_file="interview_analysis_report.html"):
    """
    분석 결과를 HTML 리포트로 저장
    
    Args:
        result (dict): 분석 결과 데이터
        output_file (str): 저장할 파일 경로
    """
    try:
        # 템플릿 파일 읽기
        with open("template.html", "r", encoding="utf-8") as f:
            template = f.read()
            
        # 데이터 준비
        analysis = result["analysis"]
        
        # 상세 내용을 HTML 리스트로 변환하는 함수
        def create_detail_list(details):
            if not details:
                return "<li>상세 내용 없음</li>"
            return "".join([f"<li>{detail}</li>" for detail in details])
            
        # 백분율을 CSS 클래스로 변환하는 함수
        def percentage_to_class(score, max_score):
            percentage = (float(score) / float(max_score)) * 100
            # 10단위로 반올림
            rounded = round(percentage / 10) * 10
            return str(int(rounded))
            
        data = {
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "question": result["question"],
            "answer": result["transcript"],
            "total_score": analysis["total_score"]["score"],
            "level": analysis["total_score"]["level"],
            "strengths": "<ul>" + "".join([f"<li>{s}</li>" for s in analysis["total_score"]["key_strengths"][:3]]) + "</ul>",
            "improvements": "<ul>" + "".join([f"<li>{i}</li>" for i in analysis["total_score"]["improvements"][:3]]) + "</ul>",
            
            # 답변 분석
            "answer_score": analysis["answer_analysis"]["subtotal"],
            "logic_score": analysis["answer_analysis"]["logic"]["score"],
            "logic_score_percent": percentage_to_class(analysis["answer_analysis"]["logic"]["score"], 5),
            "logic_details": create_detail_list(analysis["answer_analysis"]["logic"]["details"]),
            "specificity_score": analysis["answer_analysis"]["specificity"]["score"],
            "specificity_score_percent": percentage_to_class(analysis["answer_analysis"]["specificity"]["score"], 5),
            "specificity_details": create_detail_list(analysis["answer_analysis"]["specificity"]["details"]),
            "delivery_score": analysis["answer_analysis"]["delivery"]["score"],
            "delivery_score_percent": percentage_to_class(analysis["answer_analysis"]["delivery"]["score"], 5),
            "delivery_details": create_detail_list(analysis["answer_analysis"]["delivery"]["details"]),
            
            # 의도 분석
            "intent_score": analysis["intent_analysis"]["subtotal"],
            "relevance_score": analysis["intent_analysis"]["relevance"]["score"],
            "relevance_score_percent": percentage_to_class(analysis["intent_analysis"]["relevance"]["score"], 10),
            "relevance_details": create_detail_list(analysis["intent_analysis"]["relevance"]["details"]),
            "focus_score": analysis["intent_analysis"]["focus"]["score"],
            "focus_score_percent": percentage_to_class(analysis["intent_analysis"]["focus"]["score"], 5),
            "focus_details": create_detail_list(analysis["intent_analysis"]["focus"]["details"]),
            
            # 감정 분석
            "emotion_score": analysis["emotion_analysis"]["subtotal"],
            "tone_score": analysis["emotion_analysis"]["tone"]["score"],
            "tone_score_percent": percentage_to_class(analysis["emotion_analysis"]["tone"]["score"], 10),
            "sentiment": analysis["emotion_analysis"]["tone"]["sentiment"],
            "tone_details": create_detail_list(analysis["emotion_analysis"]["tone"]["details"]),
            "word_choice_score": analysis["emotion_analysis"]["word_choice"]["score"],
            "word_choice_score_percent": percentage_to_class(analysis["emotion_analysis"]["word_choice"]["score"], 5),
            "word_choice_details": create_detail_list(analysis["emotion_analysis"]["word_choice"]["details"]),
            
            # 역량 평가
            "competency_score": analysis["competency_evaluation"]["subtotal"],
            "expertise_score": analysis["competency_evaluation"]["expertise"]["score"],
            "expertise_score_percent": percentage_to_class(analysis["competency_evaluation"]["expertise"]["score"], 10),
            "expertise_keywords": create_detail_list(analysis["competency_evaluation"]["expertise"]["keywords"]),
            "expertise_details": create_detail_list(analysis["competency_evaluation"]["expertise"]["details"]),
            "communication_score": analysis["competency_evaluation"]["communication"]["score"],
            "communication_score_percent": percentage_to_class(analysis["competency_evaluation"]["communication"]["score"], 5),
            "communication_details": create_detail_list(analysis["competency_evaluation"]["communication"]["details"]),
            
            # 키워드 분석
            "positive_ratio": analysis["keyword_analysis"]["positive_ratio"],
            "professional_terms_count": analysis["keyword_analysis"]["professional_terms_count"],
            "main_keywords": create_detail_list(analysis["keyword_analysis"]["main_keywords"]),
            "emotional_expressions": create_detail_list(analysis["keyword_analysis"]["emotional_expressions"]),
            
            # 답변 통계
            "total_words": analysis["answer_statistics"]["total_words"],
            "total_sentences": analysis["answer_statistics"]["total_sentences"],
            "avg_sentence_length": analysis["answer_statistics"]["avg_sentence_length"],
            "estimated_time": analysis["answer_statistics"]["estimated_time"],
            "detail_level": analysis["answer_statistics"]["detail_level"]
        }
        
        # HTML 생성
        html_content = template.format(**data)
        
        # 파일 저장
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"HTML 리포트가 {output_file}에 저장되었습니다.")
        
    except Exception as e:
        logger.error(f"HTML 리포트 생성 중 오류 발생: {str(e)}", exc_info=True)
        raise

# ===== 5단계: 프로세스 제어 =====
# ----- 5.1 단일 면접 처리 -----
def process_interview_recording(audio_file_path, question, transcript=None):
    """
    면접 음성 파일을 텍스트로 변환하고 분석하는 메인 함수
    
    Args:
        audio_file_path (str): 분석할 음성 파일 경로
        question (str): 면접 질문
        transcript (str, optional): 이미 추출된 텍스트가 있는 경우 전달
        
    Returns:
        dict: 분석 결과를 포함하는 딕셔너리
        None: 오류 발생 시
    """
    try:
        logger.info(f"음성 파일 처리 시작: {audio_file_path}")
        
        # 5.1.1 입력 유효성 검사
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_file_path}")
            
        if not question:
            raise ValueError("질문이 입력되지 않았습니다.")
            
        # 5.1.2 음성 데이터 처리
        if transcript is None:
            logger.info("Whisper API로 음성을 텍스트로 변환 중...")
            with open(audio_file_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    language="ko",
                    response_format="text"
                )
            logger.info("음성 인식 완료")
        else:
            logger.info("이미 추출된 텍스트를 사용합니다.")
        
        # 5.1.3 텍스트 분석
        # 5.1.3.1 텍스트 구조화
        logger.info("텍스트 구조화 작업 시작...")
        structured_data = create_structured_analysis(transcript)
        if isinstance(structured_data, str) and "오류" in structured_data:
            raise Exception(structured_data)
        
        # 5.1.3.2 상세 분석
        logger.info("GPT 분석 시작...")
        analysis = analyze_interview_data(structured_data)
        
        # 5.1.4 결과 생성
        # 5.1.4.1 결과 데이터 구성
        result = {
            "question": question,
            "transcript": transcript,
            "structured_data": structured_data,
            "analysis": analysis
        }
        
        # 5.1.4.2 결과 파일 저장
        file_prefix = audio_file_path.split('.')[0]
        
        logger.info("결과 파일 저장 중...")
        # JSON 저장 (데이터 백업용)
        json_file = f"{file_prefix}_analysis_result.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # HTML 저장 (사용자 보고서용)
        html_file = f"{file_prefix}_analysis_report.html"
        save_as_html_report(result, html_file)
        
        logger.info("모든 처리가 완료되었습니다.")
        print(f"\n결과 파일:")
        print(f"1. 데이터 백업: {json_file}")
        print(f"2. 분석 리포트: {html_file}")
        
        return result
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}", exc_info=True)
        return None

# ----- 5.2 다중 면접 처리 -----
def analyze_multiple_interviews(audio_files):
    """
    여러 음성 파일을 순차적으로 분석하는 함수
    
    Args:
        audio_files (list): 분석할 음성 파일 목록
    """
    for audio_file in audio_files:
        print(f"\n{'='*50}")
        print(f"파일 분석 시작: {audio_file}")
        print('='*50)
        
        # 1. 텍스트 추출
        transcript = get_transcript_only(f"{audio_file}.wav")
        if transcript:
            print("\n[답변 내용]")
            print(transcript)
            
            # 2. 사용자에게 질문 입력 받기
            question = input("\n위 답변에 해당하는 질문을 입력해주세요: ")
            
            # 3. 전체 분석 진행 (추출된 텍스트 재사용)
            result = process_interview_recording(f"{audio_file}.wav", question, transcript)
            if result:
                print(f"\n분석이 완료되었습니다.")
            else:
                print(f"\n분석 중 오류가 발생했습니다.")
        else:
            print(f"\n텍스트 추출에 실패했습니다.")
        
        print(f"\n{'='*50}\n")

if __name__ == "__main__":
    # 테스트할 음성 파일 목록
    test_files = [
        "ckmk_a_sm_f_e_95242",
        "ckmk_a_sm_f_e_115122",
        "ckmk_a_rnd_m_n_160654",
        "ckmk_a_ict_f_n_41460",
        "ckmk_a_ard_f_e_87752"
    ]
    
    # 전체 분석 프로세스 실행
    analyze_multiple_interviews(test_files) 