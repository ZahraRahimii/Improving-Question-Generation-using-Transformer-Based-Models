from flask import Flask, request, jsonify, render_template  
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
from flask_cors import CORS


app = Flask(__name__, static_folder=os.getcwd())
CORS(app)

t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), 'model', f't5_trained_model_20'))
t5_tokenizer = T5Tokenizer.from_pretrained(os.path.join(os.path.join(os.getcwd(), 'model', f't5_tokenizer_20')))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_question():
    data = request.get_json()
    context = data['context']
    answer = data['answer']

    if not context or not answer:
        return jsonify({"error": "Both context and answer are required"}), 400

    text = f"context: {context} answer: {answer}"
    input_ids = t5_tokenizer.encode(text, return_tensors="pt", max_length=400, truncation=True)
    output_ids = t5_model.generate(input_ids=input_ids, max_length=40)
    generated_question = str(t5_tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return jsonify({"generated_question": generated_question})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
