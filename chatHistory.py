import sys
from typing import List,Dict
from ollama import Client,ResponseError
from BoundedMemoryAssistant import BoundedMemoryAssistant
from config import MODEL_NAME,DEFAULT_SYSTEM_PROMPT,TEMPERATURE,OLLAMA_HOST
from pathlib import Path

class StateFullAssistant():
    
    
    
    def __init__(self,system_prompt:str=DEFAULT_SYSTEM_PROMPT):
        self.client=Client(host=OLLAMA_HOST)
        self.system_prompt=system_prompt
        self.history: List[Dict[str, str]] = []
        self.max_turns = 5
        self.file_path=""
        self.boundedMemory = BoundedMemoryAssistant()
        self.initialize_memory(self.boundedMemory)
        
        

    def initialize_memory(self,boundedMemory:BoundedMemoryAssistant):
        print("inside initialize memory")
        data_dir=Path("./data")
        data_dir.mkdir(exist_ok=True)
        self.file_path = data_dir/"oldConversation.json"
        self.history.append({"role":"system","content":self.system_prompt})
        if self.file_path.exists():
            boundedMemory.import_session(self.history,self.file_path)
        
                
        
    def add_message(self,role:str,content:str)->None:
        self.history.append({"role":role,"content":content})
        
    def clean_memory(self)->None:
        self.history = [self.history[0]]  # Keep only the system prompt
        #self.initialize_memory()
    
    def inspect_memory(self)->None:
        """total and exact stuff stored memory"""
        print(f"Total messages in memory: {len(self.history)}")
    
    def pruneMemory(self)->None:
        self.history = self.boundedMemory._pruneMemory(self.history) 
        
    def show_conversation(self)->None:
            """Display the entire conversation history"""
            for message in self.history:
                role = message['role']
                content = message['content']
                print(f"{role}: {content}")

    def chat_stream(self,user_input:str):
        
        self.add_message("user",user_input)
        response=""
        try:
            stream=self.client.chat(
                model=MODEL_NAME,
                messages=self.history,
                stream=True, #for sending data in chunks, optimization
                options={"temperature":TEMPERATURE}
            ) 
            print("Calling LLM model ... ")
            self.pruneMemory()
            print("Assistant: ",end="", flush=True)
            for chunk in stream:
                if chunk['message'].get('content') != "":
                    token = chunk['message']['content']
                    response+=token
                    print(token,end="",flush=True)
                    #self.history.append({"role":"assistant","content":token})
            self.history.append({"role":"assistant","content":response})
            self.boundedMemory.export_session(self.history,self.file_path)
        except ResponseError as e:
            print(f"Error: {e}")
            self.history.pop()  #Rollback the last user message in case of an error
        except Exception as e:
            print(f"Exception: {e}")
            self.history.pop()  #Rollback the last user message in case of an exception 
