import os
import requests
import ollama  # DeepSeek via Ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI()

# API Configuration
BANK_API_URL = os.getenv("BANK_API_URL")  # Example: "https://api.bank.com/"
BANK_API_KEY = os.getenv("BANK_API_KEY")

# Request model
class BankQuery(BaseModel):
    account_number: str
    query_text: str  # Free text query

def classify_query_with_deepseek(query_text: str) -> str:
    """Use DeepSeek (Ollama) to classify the query into a banking category."""
    
    prompt = f"""
    Classify the following banking query into one of these categories: 
    - balance
    - mini_statement
    - last_transaction
    - loan_balance
    
    If the query doesn't match, return 'unknown'.
    
    Query: "{query_text}"
    Response:
    """

    response = ollama.chat(model="deepseek", messages=[{"role": "user", "content": prompt}])
    
    query_type = response["message"]["content"].strip().lower()
    return query_type if query_type in ["balance", "mini_statement", "last_transaction", "loan_balance"] else "unknown"

def fetch_bank_data(endpoint: str, payload: dict) -> Any:
    """Fetch data from the bank API."""
    try:
        headers = {
            "Authorization": f"Bearer {BANK_API_KEY}",
            "Content-Type": "application/json",
        }
        response = requests.post(f"{BANK_API_URL}/{endpoint}", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()  # Return raw JSON response
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"API Error: {str(e)}")

def convert_json_to_readable_format(query_text: str, raw_json: dict) -> str:
    """Use DeepSeek AI to convert JSON into a human-readable format."""
    
    prompt = f"""
    Convert the following banking API JSON response into a user-friendly message:

    Query: "{query_text}"
    JSON Response:
    {raw_json}

    Format it in a short, clear message.
    """

    response = ollama.chat(model="deepseek", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"].strip()

@app.post("/bank_query/")
def bank_query(request: BankQuery) -> Dict[str, str]:
    """Handles banking queries using AI for classification & response formatting."""
    
    # Classify the query type using DeepSeek AI
    query_type = classify_query_with_deepseek(request.query_text)
    
    endpoint_map = {
        "balance": "account/balance",
        "mini_statement": "account/mini-statement",
        "last_transaction": "account/transactions",
        "loan_balance": "account/loan-balance"
    }

    if query_type not in endpoint_map:
        raise HTTPException(status_code=400, detail="Query not recognized. Try rephrasing.")

    # Fetch raw JSON from the bank API
    raw_response = fetch_bank_data(endpoint_map[query_type], {"account_number": request.account_number})

    # Convert JSON response into human-readable format using DeepSeek AI
    readable_response = convert_json_to_readable_format(request.query_text, raw_response)
    
    return {"response": readable_response}
