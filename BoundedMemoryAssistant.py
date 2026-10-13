import sys
from ollama import Client, ResponseError
from config import DEFAULT_SYSTEM_PROMPT, MODEL_NAME, TEMPERATURE, OLLAMA_HOST
from typing import List, Dict
from pathlib import Path
import json 

class BoundedMemoryAssistant:
    
    def __init__(self):  
        self.client = Client(host=OLLAMA_HOST)
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.old_conversation:List[Dict[str,str]] = []
        self.max_turns = 5

    def _pruneMemory(self,history:List[Dict[str,str]]): #added _ infront of the function to make it private       
        if len(history) > 2 * self.max_turns:
            print("Pruning messages, saving only the last max_turn messages + initial system prompt")
            history = [history[0]] + history[-self.max_turns:] 
            print("After pruning , size of history: " + str(len(history)))
            with open("data/oldConversation.json", "w") as f:
                json.dump(history,f,indent=4)
        return history
       
    def export_session(self,history:List[Dict[str,str]],file_path:Path):
        with open(file_path, mode='w') as f :
            json.dump(history,f,indent=4)
    
    def import_session(self,history:List[Dict[str,str]],file_path:Path):
        with open(file_path, mode="r") as f :
            self.old_conversation = json.load(f) 
            history.extend(self.old_conversation)