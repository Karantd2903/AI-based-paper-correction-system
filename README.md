# AI-based-paper-correction-system
#app.py code
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load transformer model once at startup
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    model_answer = request.form['model_answer']
    student_answer = request.form['student_answer']

    # Get semantic embeddings
    embedding1 = model.encode(model_answer, convert_to_tensor=True)
    embedding2 = model.encode(student_answer, convert_to_tensor=True)

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()
    percentage = round(similarity * 100, 2)

    # Generate feedback
    feedback = "Excellent!" if percentage > 85 else (
        "Good attempt, but can be improved." if percentage > 60 else "Needs improvement.")

    return render_template("result.html", percentage=percentage, feedback=feedback)

if __name__ == '__main__':
    app.run(debug=True)
