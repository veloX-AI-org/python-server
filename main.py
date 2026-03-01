from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from Quiz import extractor, generator
from Pinecone_CRUD.main import create_index, upsert_document_data, upsert_url_content, delete_source, getContext, getSpecificContext
from getSummary.main import getResponse
from Chat.main import getChatResponse

app = FastAPI()

@app.get("/")
def home():
    return "Home"

@app.post('/generateQuiz')
async def GenerateQuize(request: Request):
    data = await request.json()
    
    # Get Video ID & Transcript
    video_id = extractor.getID(url=data['youtubeURLLink'])
    transcript = extractor.get_transript(video_id=video_id)

    # Generate Quiz
    print(f"Quiz generating of the video: {video_id}")
    quizzes = generator.generate_quiz(transcript)

    return {
        'success': True,
        'message': quizzes
    }

# verified
@app.post('/upsert_documents')
async def UpsertDocuments(request: Request):
    data = await request.json()
    
    # Create INDEX if not exist
    INDEX = create_index(data['indexID'])

    # Upsert Documents At Pinecone
    upsertedOrNot = upsert_document_data(
        docs=data['docs'], 
        DOCID=data['docID'], 
        index=INDEX
    )

    return {'message': upsertedOrNot}

# verified
@app.post('/delete_documents')
async def deleteDocuments(request: Request):
    data = await request.json()
    
    # Create INDEX if not exist
    INDEX = create_index(data['indexID'])

    # Upsert Documents At Pinecone
    deletedOrNot = delete_source(
        index=INDEX,
        docid=data['docID'],
        docType='doc'
    )

    return {'message': deletedOrNot}

# verfied
@app.post('/upsert_url_info')
async def upsert_url_info(request: Request):
    data = await request.json()
    
    # Create Index If Not Exist
    INDEX = create_index(data['indexID'])
    
    # Upsert Contect to Pinecone
    upsertedOrNot = upsert_url_content(
        url=data['url'], 
        index=INDEX, 
        docID=data['docID']
    )

    return {'message': upsertedOrNot}

# verified
@app.post('/delete_url_info')
async def deleteUrls(request: Request):
    data = await request.json()
    
    # Create INDEX if not exist
    INDEX = create_index(data['indexID'])

    # Upsert Documents At Pinecone
    deletedOrNot = delete_source(
        index=INDEX,
        docid=data['urlID'],
        docType='url'
    )

    return {'message': deletedOrNot}

# varified
@app.post("/getSummary")
async def getSummary(request: Request):
    data = await request.json()
    
    # Fetch context from pinecone database
    context = getContext(
        indexID=data['indexID'], 
        allurlIDs=data['allurlsID'], 
        alldocIDs=data['alldocsID'] 
    )

    # Feed context to model and get response
    response = getResponse(context)

    # Send response to client
    return {
        "questions": response.questions,
        "success": True,
        "summary": response.summary
    }

# verified
@app.post("/getSummaryForEveryDoc")
async def getSummaryForEveryDoc(request: Request):
    data = await request.json()
    
    context = getSpecificContext(
        sourceType = data["sourceType"],
        sourceID = data["sourceID"],
        indexID = data["indexID"]
    )

    # Feed context to model and get response
    response = getResponse(context)
    
    return {"summary": response.summary, "success": True}

# varified
@app.post("/getAIResponse")
async def getAIResponse(request: Request):
    """
    A function which reads four inputs
        - user_query: Query from user.
        - pastConverstation: Past conversation between user and ai.
        - userID: user's unique ID. and
        - notebookID

    Based on this, it first decides weather it should use a tool or simple answer.

    - If answer is not provided my model's own learning then it must be answer from context.

    And this workflow design to fetch only relevent context chunks, insuring all the docs must be equally treated.

    The document which is highly relected to a perticular source then it's document count must be higher. As compare to the other sources.
    """

    # Get data from client
    data = await request.json()
    
    if not data:
        raise HTTPException(status_code=400, detail="No JSON received")

    # Handle all the usefull parameters
    user_query = data.get("query", "")
    pastConverstation = data.get("pastConverstation", "")
    userID = data.get("userID", "")
    notebookID = data.get("notebookID", "")
    
    # Invoke our chatbot asynchronously
    response = await getChatResponse(
        query=user_query,
        past_conversation=pastConverstation,
        userID=userID,
        notebookID=notebookID
    )

    # return response
    return {"response": response['messages'][-1].content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", port=5000, reload=True)
