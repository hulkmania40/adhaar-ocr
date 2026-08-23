import os
import json
import re
import io
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from google import genai
from dotenv import load_dotenv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.5-flash-lite"

client = genai.Client(api_key=GEMINI_API_KEY)


def extract_bank_statement_data(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract transactions from bank statement image or PDF using Gemini API.
    
    Args:
        file_bytes: Binary content of the file
        filename: Original filename to determine file type
        
    Returns:
        List of transaction dictionaries
    """
    try:
        filename_lower = filename.lower()
        
        # Handle PDF files
        if filename_lower.endswith(".pdf"):
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                
                if not text or not text.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="Could not extract any text from the PDF. It may be a scanned/image-based PDF."
                    )
                
                return extract_transactions_from_text(text)
                
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="PDF processing requires pypdf. Install with: pip install pypdf"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to read PDF: {str(e)}"
                )
        
        # Handle Image files
        else:
            try:
                image = Image.open(io.BytesIO(file_bytes))
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                
                return extract_transactions_from_image(image)
                
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process image: {str(e)}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in extract_bank_statement_data: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )


def extract_transactions_from_image(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Extract transactions from a bank statement image using Gemini.
    """
    prompt = """
    You are a financial document extractor. Extract ALL bank transactions from this bank statement image.

    For each transaction, identify:
    - date: transaction date in YYYY-MM-DD format
    - description: transaction description/payee name (string)
    - amount: numeric amount (float, without currency symbols)
    - type: "credit" if money is deposited/received, "debit" if money is withdrawn/paid
    - balance: account balance after this transaction (optional, float)

    Return ONLY valid JSON — a JSON array of transaction objects:
    [
        {
            "date": "2024-01-15",
            "description": "Amazon India",
            "amount": 2500.00,
            "type": "debit",
            "balance": 42500.00
        },
        {
            "date": "2024-01-16",
            "description": "Salary Credit",
            "amount": 50000.00,
            "type": "credit",
            "balance": 92500.00
        }
    ]

    Important Rules:
    - Include EVERY transaction row you can read from the statement
    - For amount, use positive numbers for both credit and debit
    - If a field cannot be determined, use null for the value
    - DO NOT include "balance" or "closing balance" summary rows
    - DO NOT include opening balance or closing balance as transactions
    - DO NOT add any additional text or explanation outside the JSON array
    - Extract dates in YYYY-MM-DD format
    - Clean the description to remove extra whitespace and special characters
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image]
    )
    
    return parse_gemini_response(response.text)


def extract_transactions_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract transactions from plain text bank statement using Gemini.
    """
    prompt = f"""
    You are a financial document extractor. Extract ALL bank transactions from the following bank statement text.

    For each transaction, identify:
    - date: transaction date in YYYY-MM-DD format
    - description: transaction description/payee name (string)
    - amount: numeric amount (float, without currency symbols)
    - type: "credit" if money is deposited/received, "debit" if money is withdrawn/paid
    - balance: account balance after this transaction (optional, float)

    Return ONLY valid JSON — a JSON array of transaction objects:
    [
        {{
            "date": "2024-01-15",
            "description": "Amazon India",
            "amount": 2500.00,
            "type": "debit",
            "balance": 42500.00
        }},
        {{
            "date": "2024-01-16",
            "description": "Salary Credit",
            "amount": 50000.00,
            "type": "credit",
            "balance": 92500.00
        }}
    ]

    Important Rules:
    - Include EVERY transaction row you can read from the statement
    - For amount, use positive numbers for both credit and debit
    - If a field cannot be determined, use null for the value
    - DO NOT include "balance" or "closing balance" summary rows
    - DO NOT include opening balance or closing balance as transactions
    - DO NOT add any additional text or explanation outside the JSON array
    - Extract dates in YYYY-MM-DD format
    - Clean the description to remove extra whitespace and special characters

    Statement text:
    ---
    {text}
    ---
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt]
    )
    
    return parse_gemini_response(response.text)


def parse_gemini_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse Gemini response to extract JSON array.
    """
    print(f"Gemini response: {response_text[:500]}")
    
    # Try to find JSON array in the response
    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to parse the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # If JSON parsing fails, try to extract using more flexible pattern
    try:
        # Sometimes the response might have markdown code blocks
        code_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
        if code_match:
            return json.loads(code_match.group(1))
    except:
        pass
    
    raise HTTPException(
        status_code=500,
        detail="Failed to parse Gemini response as valid JSON array"
    )


@app.get("/")
async def root():
    return {
        "message": "Bank Statement Extractor API is running!",
        "model": MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY),
        "endpoints": {
            "/extract-transactions": "POST - upload an image or PDF to extract bank transactions",
            "/health": "GET - health check endpoint"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "api_key_configured": bool(GEMINI_API_KEY)
    }


@app.post("/extract-transactions")
async def extract_transactions(
    file: UploadFile = File(...)
):
    """
    Extract bank transactions from uploaded file.
    
    Supports:
    - Image files: JPEG, PNG, WebP, GIF
    - PDF files (text-based only)
    
    Returns:
    {
        "success": true,
        "model_used": "gemini-3.5-flash-lite",
        "transaction_count": 42,
        "data": [
            {
                "date": "2024-01-15",
                "description": "Amazon India",
                "amount": 2500.00,
                "type": "debit",
                "balance": 42500.00
            },
            ...
        ]
    }
    """
    # Validate content type
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type"
        )
    
    # Check allowed file types
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: images and PDF."
        )
    
    # Read file contents
    contents = await file.read()
    
    # Check file size
    max_size = 10 * 1024 * 1024  # 10MB
    if len(contents) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {max_size // (1024 * 1024)}MB limit"
        )
    
    # Extract transactions
    transactions = extract_bank_statement_data(contents, file.filename)
    
    return {
        "success": True,
        "model_used": MODEL_NAME,
        "transaction_count": len(transactions),
        "data": transactions
    }


@app.post("/extract-transactions-advanced")
async def extract_transactions_advanced(
    file: UploadFile = File(...),
    account_info: Optional[bool] = False,
    statement_period: Optional[bool] = False
):
    """
    Advanced extraction with optional account info and statement period.
    """
    # Validate content type
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type"
        )
    
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: images and PDF."
        )
    
    contents = await file.read()
    
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 10MB limit"
        )
    
    # For advanced extraction, we'll use a more comprehensive prompt
    try:
        filename_lower = file.filename.lower()
        
        if filename_lower.endswith(".pdf"):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(contents))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text or not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract any text from the PDF."
                )
            result = extract_advanced_from_text(text, account_info, statement_period)
        else:
            image = Image.open(io.BytesIO(contents))
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            result = extract_advanced_from_image(image, account_info, statement_period)
        
        return {
            "success": True,
            "model_used": MODEL_NAME,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Advanced extraction error: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Advanced extraction error: {str(e)}"
        )


def extract_advanced_from_image(image: Image.Image, include_account_info: bool, include_statement_period: bool) -> Dict:
    """
    Advanced extraction from image with optional fields.
    """
    prompt = "You are a financial document extractor. Extract ALL bank transactions from this bank statement image.\n\n"
    
    if include_account_info:
        prompt += "Also extract the following account information:\n"
        prompt += "- account_holder_name: Name of the account holder\n"
        prompt += "- account_number: Account number (masked or partial if available)\n"
        prompt += "- bank_name: Name of the bank\n"
        prompt += "- branch: Branch name or code\n\n"
    
    if include_statement_period:
        prompt += "Also extract the statement period:\n"
        prompt += "- statement_start_date: Start date of the statement period (YYYY-MM-DD)\n"
        prompt += "- statement_end_date: End date of the statement period (YYYY-MM-DD)\n\n"
    
    prompt += """
    For each transaction, identify:
    - date: transaction date in YYYY-MM-DD format
    - description: transaction description/payee name (string)
    - amount: numeric amount (float, without currency symbols)
    - type: "credit" if money is deposited/received, "debit" if money is withdrawn/paid
    - balance: account balance after this transaction (optional, float)

    Return ONLY valid JSON with this structure:
    {
        "transactions": [
            {
                "date": "2024-01-15",
                "description": "Amazon India",
                "amount": 2500.00,
                "type": "debit",
                "balance": 42500.00
            }
        ],
        "account_info": {
            "account_holder_name": "John Doe",
            "account_number": "XXXX1234",
            "bank_name": "ABC Bank",
            "branch": "Main Branch"
        },
        "statement_period": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }
    }

    Important Rules:
    - Include EVERY transaction row you can read from the statement
    - For amount, use positive numbers for both credit and debit
    - If a field cannot be determined, use null for the value
    - DO NOT include "balance" or "closing balance" summary rows
    - DO NOT add any additional text or explanation outside the JSON
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, image]
    )
    
    return parse_advanced_response(response.text)


def extract_advanced_from_text(text: str, include_account_info: bool, include_statement_period: bool) -> Dict:
    """
    Advanced extraction from text with optional fields.
    """
    prompt = "You are a financial document extractor. Extract ALL bank transactions from the following bank statement text.\n\n"
    
    if include_account_info:
        prompt += "Also extract the following account information:\n"
        prompt += "- account_holder_name: Name of the account holder\n"
        prompt += "- account_number: Account number (masked or partial if available)\n"
        prompt += "- bank_name: Name of the bank\n"
        prompt += "- branch: Branch name or code\n\n"
    
    if include_statement_period:
        prompt += "Also extract the statement period:\n"
        prompt += "- statement_start_date: Start date of the statement period (YYYY-MM-DD)\n"
        prompt += "- statement_end_date: End date of the statement period (YYYY-MM-DD)\n\n"
    
    prompt += f"""
    For each transaction, identify:
    - date: transaction date in YYYY-MM-DD format
    - description: transaction description/payee name (string)
    - amount: numeric amount (float, without currency symbols)
    - type: "credit" if money is deposited/received, "debit" if money is withdrawn/paid
    - balance: account balance after this transaction (optional, float)

    Return ONLY valid JSON with this structure:
    {{
        "transactions": [
            {{
                "date": "2024-01-15",
                "description": "Amazon India",
                "amount": 2500.00,
                "type": "debit",
                "balance": 42500.00
            }}
        ],
        "account_info": {{
            "account_holder_name": "John Doe",
            "account_number": "XXXX1234",
            "bank_name": "ABC Bank",
            "branch": "Main Branch"
        }},
        "statement_period": {{
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }}
    }}

    Important Rules:
    - Include EVERY transaction row you can read from the statement
    - For amount, use positive numbers for both credit and debit
    - If a field cannot be determined, use null for the value
    - DO NOT include "balance" or "closing balance" summary rows
    - DO NOT add any additional text or explanation outside the JSON

    Statement text:
    ---
    {text}
    ---
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt]
    )
    
    return parse_advanced_response(response.text)


def parse_advanced_response(response_text: str) -> Dict:
    """
    Parse Gemini response for advanced extraction.
    """
    print(f"Gemini response: {response_text[:500]}")
    
    # Try to find JSON object in the response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try to parse the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract from code blocks
    try:
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if code_match:
            return json.loads(code_match.group(1))
    except:
        pass
    
    raise HTTPException(
        status_code=500,
        detail="Failed to parse Gemini response as valid JSON"
    )


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🏦 Bank Statement Extractor Server")
    print("="*60)
    print(f"🤖 Using model: {MODEL_NAME}")
    print(f"🔑 API Key loaded: {'✅ Yes' if GEMINI_API_KEY else '❌ No'}")
    print("📍 Server will run at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("💳 Transaction Endpoint: POST http://localhost:8000/extract-transactions")
    print("📋 Advanced Endpoint: POST http://localhost:8000/extract-transactions-advanced")
    print("="*60)
    print("\nPress Ctrl+C to stop the server\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)