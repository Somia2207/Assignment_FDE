from rag import rag_pipeline

while True:
    print("\n")
    print("*** Your AI Assistant ***")
    print("1. Ask a query")
    print("2. Exit")
    option = input("\nUser: ")
    if option == '2':
        print("AI: Goodbye! Have a good Day!")
        break
    elif option == '1':
        user_query = input("Ask your question : ")
        print(f"AI : {rag_pipeline(user_query)}")
    else :
        print("Select the correct option!")