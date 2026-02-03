from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# A function to get all the documents of given url.
def get_document(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    splitted_document = splitter.split_documents(document)
    return [doc.page_content for doc in splitted_document]