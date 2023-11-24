 
from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the data
data = np.loadtxt('movie_reviews.txt', delimiter='\t')
X = data[:, 0]
y = data[:, 1]

# Create the vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Create the model
model = MultinomialNB()
model.fit(X, y)

# Create the Flask app
app = Flask(__name__)

# Define the routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the movie review
    review = request.form['review']

    # Vectorize the review
    X = vectorizer.transform([review])

    # Predict the sentiment
    y_pred = model.predict(X)

    # Return the results
    return jsonify({'sentiment': y_pred[0]})

@app.route('/about')
def about():
    return render_template('about.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
