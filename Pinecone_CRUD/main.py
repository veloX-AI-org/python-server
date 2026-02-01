import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# from GetDocuments import get_document
import uuid

load_dotenv()

model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding Model Loaded")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

## --------------
## CREATE INDEXES
## --------------
def create_index(index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region="us-east-1"
            )
        )

    index = pc.Index(index_name)
    return index

## --------------
## UPSERT DOCUMENTS AT PINECONE
## --------------
def upsert_document_data(docs, DOCID, index):
    try:
        embeddings = model.embed_documents(docs)
        
        vectors = []
        for i, emb in enumerate(embeddings):
            vectors.append({
                "id": f"doc-{uuid.uuid4()}",
                "values": emb,
                "metadata": {
                    "text": docs[i],
                    "source_key": DOCID
                }
            })

        index.upsert(vectors=vectors)
        return "Data Upserted Successfully!"
    except Exception as e:
        print(e)
        return "Failed to Upsert!"

## --------------
## UPSERT VALUES
## --------------
# def upsert_data(url, index):
#     try:
#         texts = get_document(url)

#         embeddings = model.embed_documents(texts)

#         vectors = []
#         for i, emb in enumerate(embeddings):
#             vectors.append({
#                 "id": f"doc-{i}",
#                 "values": emb,
#                 "metadata": {
#                     "text": texts[i],
#                     "source_url": url
#                 }
#             })

#         index.upsert(vectors=vectors)
#         return "Data Upserted Successfully!"
#     except:
#         return "Failed to Upsert!"

## --------------
## Delete QUERY
## --------------
def delete_doc(index, docid):
    try:
        print("Deleting docs with source_key =", docid)
        res = index.delete(
            filter={"source_key": {"$eq": docid}}
        )
        print("Delete response:", res)
        return "Data Deleted successfully!"
    except Exception as e:
        print("Delete failed:", e)
        return f"Failed to delete embeddings: {e}"

## --------------
## FIND QUERY
## --------------
# query_text = "What is the name of the product?"
# query_embedding = model.embed_query(query_text)

# index = pc.Index("523d9e14-2d8a-4da7-824b-4f1153d0a72d")
# results = index.query(
#     vector=query_embedding,
#     top_k=3,
#     include_metadata=True
# )

# for match in results["matches"]:
#     print(match["metadata"]["text"])

