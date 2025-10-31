"""
 CBE Career Guide Chat Backend by kings
ONLY handles chat functionality with Gemini API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import uvicorn
from datetime import datetime

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyB9i66gjoyXozmryVrLD7oPvBl9dEV9yLc"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# FastAPI app
app = FastAPI(title="CBE Chat API", version="1.0.0")

# CORS - Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    language: str = "en"

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime

# CBE System Prompt
CBE_PROMPT = """You are an AI career guidance counselor for Kenya's Competency Based Education (CBE) system.

CBE PATHWAYS:
1. STEM - Science, Technology, Engineering, Mathematics (Medicine, Engineering, Tech careers)
2. Social Sciences - Humanities, Social Studies (Law, Teaching, Journalism, Government)
3. Arts & Sports - Creative & Physical Education (Arts, Music, Sports, Entertainment)
4. Technical - Vocational & Technical Skills (Construction, Automotive, Hospitality, Trades)

KENYAN CONTEXT:
- KCSE is the main secondary exam
- University entry requires C+ minimum
- KUCCPS handles university placement
- Popular universities: University of Nairobi, JKUAT, Kenyatta University, Moi University

GUIDELINES:
- Be encouraging and supportive
- Give specific, actionable advice
- Reference Kenyan universities and job market
- Use simple language for secondary school students
- Keep responses under 300 words
- Always be positive about career prospects

Answer the student's question about CBE pathways and careers in Kenya."""

@app.get("/")
async def root():
    """API status"""
    return {"message": "CBE Chat API is running", "status": "active"}

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/debug")
async def debug():
    """Debug endpoint to check configuration"""
    return {
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "gemini_api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY else 0,
        "endpoints": ["/", "/health", "/chat", "/debug"],
        "cors_enabled": True
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages"""
    try:
        print(f"📨 Received chat request: {request.message[:50]}...")
        print(f"🌍 Language: {request.language}")
        
        # Prepare language instruction
        language_instruction = "Respond in English." if request.language == "en" else "Respond in Kiswahili."
        
        # Create full prompt
        full_prompt = f"{CBE_PROMPT}\n\n{language_instruction}\n\nStudent Question: {request.message}"
        print(f"🤖 Calling Gemini API...")
        
        # Prepare Gemini API request
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 512,
            }
        }
        
        # Call Gemini API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="AI service error")
            
            result = response.json()
            
            # Extract response
            if "candidates" in result and len(result["candidates"]) > 0:
                ai_response = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"✅ Got Gemini response: {len(ai_response)} characters")
                
                return ChatResponse(
                    response=ai_response,
                    timestamp=datetime.now()
                )
            else:
                print(f"❌ Invalid Gemini response: {result}")
                raise HTTPException(status_code=500, detail="Invalid AI response")
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting CBE Chat Backend...")
    print("📍 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("💬 Chat: POST http://localhost:8000/chat")
    print("✅ Ready for frontend!")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)