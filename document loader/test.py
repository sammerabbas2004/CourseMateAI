from langchain_community.document_loaders import TextLoader

# from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("D:\\Artificial Intelligence Completee Corse\\Python\\Course Mate AI Rag\\document loader\\notes.txt")

# splitter = CharacterTextSplitter(
#     separator="",
#     chunk_size=10,
#     chunk_overlap=1
# )

doc=data.load()

# chunks=splitter.split_documents(doc)

# for i in chunks:
#     print(i.page_content)
#     print()
#     print()
#     print()   