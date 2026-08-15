from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

# Check if API key is loaded
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print(f"✅ API Key loaded successfully!")
    print(f"First 10 chars: {api_key[:10]}...")
    print(f"Length: {len(api_key)} characters")
else:
    print("❌ API Key NOT found in .env file!")
    print("Please create .env file with: GEMINI_API_KEY=your_key_here")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")