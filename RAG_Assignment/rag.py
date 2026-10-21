import os
from pathlib import Path
from ollama import Client
import numpy as np

client = Client(host="http://localhost:11434")

#Load documents from an external folder and store in knowledge base documents []
def load_KB_from_docs() :
    documents = []
    folder = Path("./RAG_Assignment/knowledgeBase")
    for count, file in enumerate(folder.glob("*.txt"), start=1):
        documents.append({
            "id": count,
            "fileName": file.name,
            "text": file.read_text(encoding="utf-8"),
        })
    
    return documents

def getEmbedding(query):   
    response = client.embeddings(model="nomic-embed-text", prompt=query)
    return response["embedding"]

def cosine_similarity(vect1, vect2):
    v1,v2 = np.array(vect1), np.array(vect2)
    dot_product = np.dot(v1,v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product/(norm1*norm2)

def semantic_search(user_query,docs,top_k = 1):
    query_embedding = getEmbedding(user_query)
    for doc in docs : 
        doc["embedding"] = getEmbedding(doc["text"])
        doc["score"] = cosine_similarity(query_embedding,doc["embedding"])    
    sorted_docs=sorted(docs,key=lambda x:x["score"],reverse=True)
    return sorted_docs[:top_k]

def rag_pipeline(user_query):
    docs = load_KB_from_docs()  
    retrieved_docs=semantic_search(user_query,docs)
    best_context=retrieved_docs[0]["text"]
    print("============================================")
    print(f"Retrieved Context from Knowledge Base: {best_context}")
    print("============================================")
    
    augmented_prompt=f""" Answer the question based ONLY on the following context.
    If the answer is not in the context, say I don't know."
    Context:{best_context}
    Question:{user_query}
    """      
    response=client.chat(
            model="qwen3:4b",
            messages=[{"role":"user","content":augmented_prompt}]
        )
    return response['message']['content']

