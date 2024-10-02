from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from src.helpers.index import langchain_helper as chatbot_helper
from src.controllers.causality import causality
import src.helpers.index as helper


helper.initializaion()
print("Done till 1")
gemini_embeddings, model =helper.models()
print("Done till 2")

retriever_prompt = (
    "Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
)

ensemble_retriever,contextualize_q_prompt = chatbot_helper.context_prompt(retriever_prompt,gemini_embeddings)
print("done till 3")
history_aware_retriever1 = create_history_aware_retriever(model,ensemble_retriever,contextualize_q_prompt)
print("fucked here")

system_prompt = (
    "You are an assistant for generating responses." 
    "Use the following pieces of retrieved context to generate response."
    "you can also use your own information in combination to help generate response."
    "Provide the answer in about 150-200 words"
    "Provide the response in great detail."
    "{context}"
)

question_answer_chain = chatbot_helper.qa_chain(system_prompt,model)
print("fucked 5")
rag_chain = create_retrieval_chain(history_aware_retriever1, question_answer_chain)
print("FUCK UUU")
chat_history = []

def generate_response(data):
    result,graph = causality(data)
    session_id = data['session_id']

    question = f"Based on the graph structure : {graph} and the result obtained : {result} through the model provide a monthly plan to improvise the condition"
    message1= chatbot_helper.conversational_rag(rag_chain).invoke(
        {"input": question},
        config={
            "configurable": {"session_id": f"{session_id}"}
        },  # constructs a key "abc123" in `store`.
    )
    print("message",message1['answer'])
    return {"result":message1["answer"]}, 200

def generate_chat(data):
    question = data['question']
    session_id = data['session_id']
    print("Helloe")
    print(question)
    message1= chatbot_helper.conversational_rag(rag_chain).invoke(
        {"input": question},
        config={
            "configurable": {"session_id": f"{session_id}"}
        },  # constructs a key "abc123" in `store`.
    )
    print("message",message1['answer'])
    return {"result":message1["answer"]}, 200

