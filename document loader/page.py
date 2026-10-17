# WebBaseLoader is a base class for loading documents from the web. It provides a common interface for different web document loaders, allowing you to easily load and process web content.

from langchain_community.document_loaders import WebBaseLoader
url="https://docs.langchain.com/oss/python/langchain/install"
data = WebBaseLoader(url)
doc = data.load()
print(doc[0].page_content)