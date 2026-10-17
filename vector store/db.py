from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
from langchain_core.documents import Document
docs = [
    Document(page_content="Python is used for data science.", metadata={"source": "doc1"}),
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "doc2"}),
    Document(page_content="Data science is an interdisciplinary field.", metadata={"source": "doc3"})
]
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="Chroma-db"
)
result = vectorstore.similarity_search("What is  used in data science?", k=2)
for r in result:
    print(r.page_content)

# retriver=vectorstore.as_retriever()