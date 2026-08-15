import os
import json
import re
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not found in .env file!")
    exit(1)

print(f"✅ GEMINI_API_KEY loaded successfully")

genai.configure(api_key=GEMINI_API_KEY)

# Use a model from your available list
# Based on your available models, these are good options for Aadhaar extraction:
MODEL_NAME = "models/gemini-3.5-flash-lite"

print(f"\n🤖 Using model: {MODEL_NAME}")

try:
    model = genai.GenerativeModel(MODEL_NAME)
    print("✅ Model initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    exit(1)

def extract_aadhaar_data(image_bytes):
    """Extract Aadhaar details using Gemini API"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Resize image if too large
        max_size = (1024, 1024)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Prepare prompt
        prompt = """
        Extract the following details from this Aadhaar card image:
        1. Name (Full name)
        2. Aadhaar Number (12-digit number)
        3. Date of Birth (in DD/MM/YYYY format)
        4. Gender (Male/Female/Other)
        5. Address (Complete address)
        
        Return ONLY valid JSON format with these exact keys:
        {
            "name": "",
            "aadhaar_number": "",
            "date_of_birth": "",
            "gender": "",
            "address": ""
        }
        If any field is not found, leave it as empty string.
        Do not add any additional text or explanation.
        """
        
        # Generate response
        response = model.generate_content([prompt, image])
        
        # Parse response
        response_text = response.text
        print(f"Raw response: {response_text[:200]}...")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            extracted_data = json.loads(json_str)
        else:
            extracted_data = json.loads(response_text)
        
        return extracted_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/extract-aadhaar")
async def extract_aadhaar(file: UploadFile = File(...)):
    """Endpoint to extract Aadhaar details from uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file contents
        contents = await file.read()
        
        # Validate file size (max 5MB)
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")
        
        # Extract data using Gemini
        extracted_data = extract_aadhaar_data(contents)
        
        return JSONResponse(
            content={
                "success": True,
                "model_used": MODEL_NAME,
                "data": extracted_data
            }
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/available-models")
async def get_available_models():
    """
    List all available Gemini models that support generateContent
    """
    try:
        models_list = []
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                models_list.append({
                    "name": model.name,
                    "display_name": model.display_name,
                    "description": model.description[:200] + "..." if len(model.description) > 200 else model.description,
                    "supported_methods": model.supported_generation_methods,
                })
        
        return {
            "success": True,
            "total_models": len(models_list),
            "current_model": MODEL_NAME,
            "models": models_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Aadhaar Extractor API is running!",
        "model_used": MODEL_NAME,
        "api_key_loaded": bool(GEMINI_API_KEY),
        "endpoints": {
            "POST /extract-aadhaar": "Upload Aadhaar image for extraction",
            "GET /available-models": "List all available Gemini models",
            "GET /health": "Health check",
            "GET /docs": "Swagger documentation"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting Aadhaar Extractor Server...")
    print("="*60)
    print(f"🤖 Using model: {MODEL_NAME}")
    print(f"🔑 API Key loaded: {'✅ Yes' if GEMINI_API_KEY else '❌ No'}")
    print("📍 Server will run at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("📋 List Models: http://localhost:8000/available-models")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)