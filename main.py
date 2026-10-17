from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint,HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

#embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#vectorstore
vectorstore = Chroma(
    persist_directory = "Chroma-db",
    embedding_function=embedding_model,
)
#Retriever
retriever=vectorstore.as_retriever(
    
    search_type="mmr",
    search_kwargs={"k":3,"fetch_k":10,"lambda_mult":0.5}
)
#large language model

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    
)
model=ChatHuggingFace(llm=llm)
#Prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.you will be provided with context and a question. Answer the question based on the context. If the question can't be answered based on the context, say \"I don't know\"."),
        ("human","Answer the question based on the context below. If the question can't be answered based on the context, say \"I don't know\".\n\nContext: {context}\n\nQuestion: {question}")
    ]
)
print("Rag System Created")
print("Press 0 to exit")
while True:
    query=input("You: ")
    if query=="0":
        break
    docs=retriever.invoke(query)
    context="".join([doc.page_content for doc in docs])
    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })
    response=llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")