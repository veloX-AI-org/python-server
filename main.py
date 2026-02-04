from flask import Flask, request, jsonify 
from Quiz import extractor, generator
from Pinecone_CRUD.main import create_index, upsert_document_data, upsert_url_content, delete_source

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

if __name__ == "__main__":
    app.run(debug=True)
