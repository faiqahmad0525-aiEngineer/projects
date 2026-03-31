from transformers import pipeline

# Load NLP model
qa_pipeline = pipeline("question-answering")

# 📚 Knowledge Base (Your Data)
context_data = """
Interns can apply leave through the HR portal.
Working hours are from 9 AM to 5 PM.
Interns must submit weekly reports every Friday.
For technical issues, contact IT support.
Internship duration is 3 months.
Stipend is paid at the end of each month.
"""

# 🤖 Chatbot Function
def chatbot():
    print("🤖 Internship Support Chatbot (type 'exit' to stop)\n")
    
    while True:
        question = input("You: ")
        
        if question.lower() == "exit":
            print("Bot: Goodbye 👋")
            break
        
        # Get answer from model
        result = qa_pipeline(question=question, context=context_data)
        
        print("Bot:", result['answer'])

# Run chatbot
chatbot()
