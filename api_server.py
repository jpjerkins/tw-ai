from flask import Flask, request, jsonify
from flask_cors import CORS
from tiddlywiki_api import answer_question_with_tiddlers
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/ask', methods=['POST'])
def ask():
    """
    Handle ask queries from the frontend.

    Expected JSON payload:
    {
        "question": "Your question here",
        "top_k": 5 (optional, default: 5),
        "model": "gpt-4o-mini" (optional, default: gpt-4o-mini)
    }

    Returns:
    {
        "question": "The original question",
        "answer": "The generated answer",
        "sources": [
            {
                "title": "Tiddler title",
                "link_url": "URL to tiddler",
                "rank": 0.95
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'Missing required field: question'}), 400

        question = data['question']
        top_k = data.get('top_k', 5)
        model = data.get('model', 'gpt-4o-mini')

        # Validate parameters
        if not isinstance(question, str) or not question.strip():
            return jsonify({'error': 'Question must be a non-empty string'}), 400

        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
            return jsonify({'error': 'top_k must be an integer between 1 and 50'}), 400

        # Call the existing function
        result = answer_question_with_tiddlers(
            question=question,
            top_k=top_k,
            model=model
        )

        return jsonify(result), 200

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"Starting TiddlyWiki AI API server on port {port}...")
    print(f"Make sure the database is running and indexed!")

    app.run(host='0.0.0.0', port=port, debug=debug)
