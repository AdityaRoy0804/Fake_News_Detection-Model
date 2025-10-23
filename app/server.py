from flask import Flask, request, jsonify
# The config will be loaded automatically when the classifier is imported.
from app.classifier import get_verifier

app = Flask(__name__)

# Set a custom error handler for 500 Internal Server Error
@app.errorhandler(500)
def internal_error(error):
    # This provides a more informative JSON response on internal errors
    return jsonify({"error": "Internal Server Error", "detail": str(error)}), 500

@app.route("/classify", methods=["POST"])
def classify_item():
    """
    Receives a JSON payload with 'text' and classifies it.
    """
    if not request.json or "text" not in request.json:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    news_item = request.json
    text_to_classify = news_item.get("text")
    item_id = news_item.get("id")

    # The get_verifier() function handles loading the model on the first call
    verifier = get_verifier()
    
    # The model inference is a blocking call, but Flask handles it in a worker thread
    result = verifier.classify(text_to_classify)

    return jsonify({
        "id": item_id,
        "input": text_to_classify,
        "result": result
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # For development only. For production, use a WSGI server like Gunicorn.
    # Example: gunicorn --workers 4 --threads 2 --bind 0.0.0.0:8000 app.server:app
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
