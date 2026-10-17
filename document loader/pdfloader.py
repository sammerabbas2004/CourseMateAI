from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data =PyPDFLoader("D:\\Artificial Intelligence Completee Corse\\Python\\Course Mate AI Rag\\document loader\\GRU.pdf")

doc=data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
chunks=splitter.split_documents(doc)
# print(len(doc))
print(chunks[0].page_content)
# print(doc[0].page_content)