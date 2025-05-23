from fastapi import FastAPI, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
from datetime import datetime
import json
import asyncio
from typing import Optional
import logging

# 기존 분석 함수들 import
from interview_analyzer import process_interview_recording, get_transcript_only, save_as_html_report

app = FastAPI(title="면접 분석 서비스", description="AI 기반 면접 분석 플랫폼")

# 정적 파일 및 템플릿 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# 업로드 디렉토리 생성
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 분석 진행 상태를 저장할 딕셔너리
analysis_status = {}

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    """메인 페이지"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_audio(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    question: str = Form(...)
):
    """음성 파일 업로드 및 분석 시작"""
    
    # 고유 ID 생성
    analysis_id = str(uuid.uuid4())
    
    # 파일 저장
    file_extension = audio_file.filename.split('.')[-1]
    file_path = f"uploads/{analysis_id}.{file_extension}"
    
    with open(file_path, "wb") as buffer:
        content = await audio_file.read()
        buffer.write(content)
    
    # 분석 상태 초기화
    analysis_status[analysis_id] = {
        "status": "uploaded",
        "progress": 0,
        "message": "파일 업로드 완료",
        "timestamp": datetime.now().isoformat()
    }
    
    # 백그라운드에서 분석 시작
    background_tasks.add_task(analyze_audio, analysis_id, file_path, question)
    
    return JSONResponse({
        "analysis_id": analysis_id,
        "message": "분석이 시작되었습니다."
    })

async def analyze_audio(analysis_id: str, file_path: str, question: str):
    """백그라운드에서 음성 분석 수행"""
    try:
        # 1단계: 음성 인식
        analysis_status[analysis_id].update({
            "status": "transcribing",
            "progress": 25,
            "message": "음성을 텍스트로 변환 중..."
        })
        
        transcript = get_transcript_only(file_path)
        if not transcript:
            raise Exception("음성 인식에 실패했습니다.")
        
        # 2단계: GPT 분석
        analysis_status[analysis_id].update({
            "status": "analyzing",
            "progress": 50,
            "message": "AI 분석 진행 중..."
        })
        
        result = process_interview_recording(file_path, question, transcript)
        if not result:
            raise Exception("분석에 실패했습니다.")
        
        # 3단계: 결과 저장
        analysis_status[analysis_id].update({
            "status": "saving",
            "progress": 75,
            "message": "결과 저장 중..."
        })
        
        # 결과 파일 저장
        result_path = f"results/{analysis_id}_result.json"
        html_path = f"results/{analysis_id}_report.html"
        
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # HTML 리포트 생성
        save_as_html_report(result, html_path)
        
        # 완료
        analysis_status[analysis_id].update({
            "status": "completed",
            "progress": 100,
            "message": "분석 완료",
            "result_path": result_path,
            "html_path": html_path
        })
        
        # 임시 파일 정리
        os.remove(file_path)
        
    except Exception as e:
        analysis_status[analysis_id].update({
            "status": "error",
            "progress": 0,
            "message": f"오류 발생: {str(e)}"
        })
        logging.error(f"Analysis error for {analysis_id}: {e}")

@app.get("/status/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """분석 진행 상태 조회"""
    if analysis_id not in analysis_status:
        return JSONResponse({"error": "분석 ID를 찾을 수 없습니다."}, status_code=404)
    
    return JSONResponse(analysis_status[analysis_id])

@app.get("/result/{analysis_id}", response_class=HTMLResponse)
async def get_analysis_result(request: Request, analysis_id: str):
    """분석 결과 페이지"""
    if analysis_id not in analysis_status:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "분석 ID를 찾을 수 없습니다."
        })
    
    status = analysis_status[analysis_id]
    if status["status"] != "completed":
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "분석이 아직 완료되지 않았습니다."
        })
    
    # 결과 파일 로드
    try:
        with open(status["result_path"], "r", encoding="utf-8") as f:
            result = json.load(f)
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "result": result,
            "analysis_id": analysis_id
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": f"결과를 불러오는 중 오류가 발생했습니다: {str(e)}"
        })

@app.get("/download/{analysis_id}")
async def download_result(analysis_id: str):
    """분석 결과 HTML 다운로드"""
    if analysis_id not in analysis_status:
        return JSONResponse({"error": "분석 ID를 찾을 수 없습니다."}, status_code=404)
    
    status = analysis_status[analysis_id]
    if status["status"] != "completed":
        return JSONResponse({"error": "분석이 완료되지 않았습니다."}, status_code=400)
    
    try:
        html_path = status["html_path"]
        if os.path.exists(html_path):
            return FileResponse(
                html_path, 
                media_type="text/html",
                filename=f"interview_analysis_report_{analysis_id}.html"
            )
        else:
            return JSONResponse({"error": "HTML 파일을 찾을 수 없습니다."}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": f"파일 다운로드 실패: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 