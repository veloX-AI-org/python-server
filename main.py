import time
from flask import Flask, request, jsonify
from Quiz import extractor, generator
from Pinecone_CRUD.main import create_index, upsert_document_data, upsert_url_content, delete_source, getContext, getSpecificContext
from getSummary.main import getResponse
from Chat.main import getChatResponse
import asyncio

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Home"

@app.route('/generateQuiz', methods=['POST'])
def GenerateQuize():
    data = request.get_json()
    
    # Get Video ID & Transcript
    video_id = extractor.getID(url=data['youtubeURLLink'])
    transcript = extractor.get_transript(video_id=video_id)

    # Generate Quiz
    print(f"Quiz generating of the video: {video_id}")
    quizzes = generator.generate_quiz(transcript)

    return jsonify({
        'success': True,
        'message': quizzes
    })

@app.route('/upsert_documents', methods=['POST'])
def UpsertDocuments():
    data = request.get_json()
    
    # Create INDEX if not exist
    INDEX = create_index(data['indexID'])

    # Upsert Documents At Pinecone
    upsertedOrNot = upsert_document_data(
        docs=data['docs'], 
        DOCID=data['docID'], 
        index=INDEX
    )

    return jsonify({
        'message': upsertedOrNot
    })

@app.route('/delete_documents', methods=['POST'])
def deleteDocuments():
    data = request.get_json()
    
    # Create INDEX if not exist
    INDEX = create_index(data['indexID'])

    # Upsert Documents At Pinecone
    deletedOrNot = delete_source(
        index=INDEX,
        docid=data['docID'],
        docType='doc'
    )

    return jsonify({
        'message': deletedOrNot
    })

@app.route('/upsert_url_info', methods=['POST'])
def upsert_url_info():
    data = request.get_json()
    
    # Create Index If Not Exist
    INDEX = create_index(data['indexID'])
    
    # Upsert Contect to Pinecone
    upsertedOrNot = upsert_url_content(
        url=data['url'], 
        index=INDEX, 
        docID=data['docID']
    )

    return jsonify({
        'message': upsertedOrNot
    })

@app.route('/delete_url_info', methods=['POST'])
def deleteUrls():
    data = request.get_json()
    
    # Create INDEX if not exist
    INDEX = create_index(data['indexID'])

    # Upsert Documents At Pinecone
    deletedOrNot = delete_source(
        index=INDEX,
        docid=data['urlID'],
        docType='url'
    )

    return jsonify({
        'message': deletedOrNot
    })

@app.route("/getSummary", methods=['POST'])
def getSummary():
    data = request.get_json()
    
    # Fetch context from pinecone database
    context = getContext(
        indexID=data['indexID'], 
        allurlIDs=data['allurlsID'], 
        alldocIDs=data['alldocsID']
    )

    # Feed context to model and get response
    response = getResponse(context)

    # Send response to client
    return jsonify({
        "questions": response.questions,
        "success": True,
        "summary": response.summary
    })

@app.route("/getSummaryForEveryDoc", methods=['POST'])
def getSummaryForEveryDoc():
    data = request.get_json()
    
    context = getSpecificContext(
        sourceType = data["sourceType"],
        sourceID = data["sourceID"],
        indexID = data["indexID"]
    )

    # Feed context to model and get response
    response = getResponse(context)
    
    return jsonify({
        "summary": response.summary,
        "success": True
    })

@app.route("/getAIResponse", methods=['POST'])
def getAIResponse():
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
    data = request.get_json()
    
    if not data:
        return {"error": "No JSON received"}, 400

    # Handle all the usefull parameters
    user_query = data.get("query", "")
    pastConverstation = data.get("pastConverstation", "")
    userID = data.get("userID", "")
    notebookID = data.get("notebookID", "")
    
    # Invoke our chatbot asynchronously
    response = asyncio.run(getChatResponse(
        query=user_query,
        past_conversation=pastConverstation,
        userID=userID,
        notebookID=notebookID
    ))

    # return response
    return jsonify({
        "response": response['messages'][-1].content
    })

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
