from flask import Flask, render_template, request, jsonify
import pickle

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Sentiment map to convert labels to human-readable sentiment
sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Route to render the input form (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the review text from the form
        review_text = request.form['review_text']

        # Preprocess and vectorize the input
        vectorized_input = vectorizer.transform([review_text])

        # Predict sentiment
        prediction = model.predict(vectorized_input)[0]
        confidence = model.predict_proba(vectorized_input).max()

        # Get the corresponding sentiment label
        sentiment = sentiment_map[prediction]

        # Render the result page with prediction
        return render_template(
            'result.html',
            review_text=review_text,
            sentiment=sentiment,
            confidence_score=round(confidence, 4)
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

