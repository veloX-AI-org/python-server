import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from .GetDocuments import get_document
import uuid

load_dotenv()

model = OpenAIEmbeddings(
    model="text-embedding-3-small"  
)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

## --------------
## CREATE INDEXES
## --------------
def create_index(index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)
    return index

## --------------
## UPSERT DOCUMENTS
## --------------
def upsert_document_data(docs, DOCID, index):
    try:
        print("Upserting document with source_key =", DOCID)
        embeddings = model.embed_documents(docs)

        vectors = []
        for i, emb in enumerate(embeddings):
            if not emb or not isinstance(emb, list):
                print(f"Skipping doc {i}: missing embedding")
                continue

            vectors.append({
                "id": f"doc-{uuid.uuid4()}",
                "values": emb,
                "metadata": {
                    "text": docs[i],
                    "source_key": DOCID
                }
            })

        if not vectors:
            print("No valid vectors to upsert")
            return "We failed to get information from profived file. Nothing to upsert"

        res = index.upsert(vectors=vectors)
        print("Upsert response:", res)
        return "Data Upserted Successfully!"

    except Exception as e:
        print("Upsert failed:", e)
        return "Failed"

## --------------
## UPSERT VALUES
## --------------
def upsert_url_content(url, index, docID):
    try:
        print("Upserting url with source_urlID =", docID)
        texts = get_document(url)
        embeddings = model.embed_documents(texts)

        vectors = []
        for i, emb in enumerate(embeddings):
            vectors.append({
                "id": f"doc-{i}",
                "values": emb,
                "metadata": {
                    "text": texts[i],
                    "source_urlID": docID
                }
            })

        res = index.upsert(vectors=vectors)
        print("Upsert response:", res)
        return "Data Upserted Successfully!"
    except:
        return "Failed to Upsert!"        

## -------------
## Delete URLs and Documents
## -------------    
def delete_source(index, docid, docType):
    try:
        print("Deleting docs with source_key =", docid)
        if docType == 'doc':
            res = index.delete(
                filter={"source_key": {"$eq": docid}}
            )
        else:
            res = index.delete(
                filter={"source_urlID": {"$eq": docid}}
            )
        print("Delete response:", res)
        return "Data Deleted successfully!"
    except Exception as e:
        print("Delete failed:", e)
        return f"Failed to delete embeddings: {e}"

## --------------
## FIND SUMMARY QUERY
## --------------
def getContext(indexID, allurlIDs, alldocIDs):
    final_context = ""

    query_text = "general information in the document" # General query for all document to get summary. 
    # Generate embeddings of above query using GPT embedding model.
    query_embedding = model.embed_query(query_text)

    # Index always exits
    index = pc.Index(indexID)

    final_context += "URL Content: \n"
    if (allurlIDs):
        for urlID in allurlIDs:
            # Get 5 context infomation object for every url information
            results = index.query(
                vector=query_embedding,
                top_k=5,
                filter = { 
                    'source_urlID': urlID
                },
                include_metadata=True
            )

            # Return text from those objects
            final_context += '\n'.join([docs['metadata']['text'] for docs in results.matches])

    final_context += "\n==============\n"
    final_context += "DOCUMENT Content: \n"

    if (alldocIDs):
        for docid in alldocIDs:
            # Get 2 context infomation object for every docs information
            results = index.query(
                vector=query_embedding,
                top_k=3,
                filter = { 
                    'source_key': docid
                },
                include_metadata=True
            )

            # Return text from those objects
            final_context += '\n'.join([docs['metadata']['text'] for docs in results.matches])

    return final_context
