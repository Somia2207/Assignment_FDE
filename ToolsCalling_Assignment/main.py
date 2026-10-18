import os
from ollama import Client
from tools import FILE_EXPLAIN_TOOL_SCHEMA, FILE_SUMMARIZER_TOOL_SCHEMA,ASK_TOOL_SCHEMA,TOOL_REGISTRY


roles = {
    "1": "You are a skilled summary writer. Summarize the given document notes.txt",

    "2": "You are a senior Python developer. Explain in short the programming concepts from notes.txt with examples",

    "3": "You are a professional python developer. Answer the given question strictly based on the given File : notes.txt and Question: "
}
while True:
    print("\n")
    print("Enter your choice")
    print("1. Summarize the document")
    print("2. Explain the document")
    print("3. Ask Questions based on the document")
    print("4. Exit")
    choice = input("Your Choice : ")
    content = roles.get(
        choice,
        "You are a helpful AI assistant.")
    if choice == '4':
        break
    if choice == '3' : 
        question = input("Ask your question ? \n")
        content = content + question

    client = Client(host="http://localhost:11434")
    try:
        response=client.chat(
            model="qwen3:4b",
            messages=[{"role":"user","content":content}],
            tools=[FILE_EXPLAIN_TOOL_SCHEMA,FILE_SUMMARIZER_TOOL_SCHEMA,ASK_TOOL_SCHEMA]
        )
    except Exception as e:
        print("Exception: " , e)

    if response.get("message",{}).get("tool_calls"):
        for tool in response["message"]["tool_calls"]:
            function_name=tool["function"]["name"] 
            arguments=tool["function"]["arguments"] 

            # execute the function matching the name
            tool_func=TOOL_REGISTRY[function_name]
            print(f"Tool/Fucntion name chosen by LLM : {function_name} with arguments : {arguments}" )
            tool_result=tool_func(**arguments)

            print(tool_result)