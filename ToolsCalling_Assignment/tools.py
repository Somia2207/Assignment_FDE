import os
from ollama import Client

client = Client(host="http://localhost:11434")

WORKSPACE_DIR = "./data"
def read_text_file(filename:str)->str:
    safe_path=os.path.abspath(os.path.join(WORKSPACE_DIR,filename))
    if not os.path.exists(safe_path):
        return f"Error: File {filename} not found!"
    with open(safe_path,"r") as f:
        return f.read()
    
def summarize_text_file(filename:str)->str:
    document = read_text_file(filename)
    prompt = f"Summarize the given document in 3 lines{document}"
    
    try:
        response=client.chat(
            model="qwen3:4b",
            messages=[{"role":"user","content":prompt}]
        )
    except Exception as e:
        print("Exception: " , e)
    return response['message']['content']

def explain_text_file(filename:str)->str:
    document = read_text_file(filename)
    prompt = f"Explain the given document : {document}"
    try:
        response=client.chat(
            model="qwen3:4b",
            messages=[{"role":"user","content":prompt}]
        )
    except Exception as e:
        print("Exception: " , e)
    return response['message']['content']

def ask_from_text_file(filename:str,question:str)->str:
    document = read_text_file(filename)
    prompt = f"""
        You are given a document. Answer the user's question using only the information present in the document.
        If the answer is not available,say: 'I couldn't find that information in the document.'
        Document: {document}
        Question: {question} """
    try:
        response=client.chat(
            model="qwen3:4b",
            messages=[{"role":"user","content":prompt}]
        )
    except Exception as e:
        print("Exception: " , e)
    return response['message']['content']
    
FILE_SUMMARIZER_TOOL_SCHEMA={
    "type":"function",
    "function":{
        "name":"summarize_text_file",
        "description":"Read and return the 3 line summary of the document notes.txt",
       "parameters": {
                   "type":"object",
                   "properties": {
                       "filename":{
                           "type":"string",
                           "description": "Name of the file"
                       }
                   },"required":["filename"]
        }
    }    
}

FILE_EXPLAIN_TOOL_SCHEMA={
    "type":"function",
    "function":{
        "name":"explain_text_file",
        "description":"Explain the document and return the text",
        "parameters": {
            "type":"object",
            "properties": {
                "filename":{
                    "type":"string",
                    "description": "Name of the file"
                }
            },"required":["filename"]
        }
        
    }    
}

ASK_TOOL_SCHEMA={
    "type":"function",
    "function":{
        "name":"ask_from_text_file",
        "description":"Ask question based on the given document and return the answer if found ",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Exact name of the file mentioned by the user"
                },
                "question": {
                    "type": "string",
                    "description": "The question to answer from the file"
                }
            }
        ,"required":["filename","question"]
        }
    }  
}

TOOL_REGISTRY={
    "explain_text_file":explain_text_file,
    "summarize_text_file":summarize_text_file,
    "ask_from_text_file":ask_from_text_file
}