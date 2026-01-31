from flask import Flask, request, jsonify
from Quiz import extractor, generator
from Pinecone_CRUD.main import create_index, upsert_document_data

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Home"

# @app.route('/generateQuiz', methods=['POST'])
# def GenerateQuize():
#     data = request.get_json()
    
#     # Get Video ID & Transcript
#     video_id = extractor.getID(url=data['youtubeURLLink'])
#     transcript = extractor.get_transript(video_id=video_id)

#     # Generate Quiz
#     quizzes = generator.generate_quiz(transcript)

#     return jsonify({
#         'success': True, 
#         'message': quizzes
#     })

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
        'success': True,
        'message': upsertedOrNot
    })

if __name__ == "__main__":
    app.run(debug=True)
