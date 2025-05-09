import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import pytesseract
import spacy
import language_tool_python
from sentence_transformers import SentenceTransformer


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True



nlp = spacy.load('en_core_web_sm')
language_tool = language_tool_python.LanguageTool('en-US')
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')


UPLOADS_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
MODEL_ANSWER_FILE = 'model_answer.txt'


os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)



def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
    elif ext in ['.txt']:
        with open(file_path, 'r') as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format!")
    return text

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def calculate_keyword_match(student_text, model_answer):
    student_words = set(student_text.split())
    model_words = set(model_answer.split())
    common_words = student_words.intersection(model_words)
    return len(common_words), len(model_words)

def check_grammar(text):
    matches = language_tool.check(text)
    grammar_score = max(0, 100 - len(matches)) 
    return grammar_score

def calculate_answer_length_score(student_answer, model_answer):
    
    student_word_count = len(student_answer.split())
    model_word_count = len(model_answer.split())
    
   
    if abs(student_word_count - model_word_count) <= 10:
        return 100  
    elif abs(student_word_count - model_word_count) <= 20:
        return 80 
    else:
        return 60  

def evaluate_assignment(file_name, total_marks):
    file_path = os.path.join(UPLOADS_FOLDER, file_name)
    student_text = extract_text_from_file(file_path)
    student_text_processed = preprocess_text(student_text)

 
    with open(MODEL_ANSWER_FILE, 'r') as f:
        model_answer = f.read()
    model_answer_processed = preprocess_text(model_answer)

 
    keyword_matches, total_keywords = calculate_keyword_match(student_text_processed, model_answer_processed)
    keyword_match_percentage = (keyword_matches / total_keywords) * 100 if total_keywords > 0 else 0
    
  
    grammar_score = check_grammar(student_text)

  
    answer_length_score = calculate_answer_length_score(student_text, model_answer)

  
    final_score = (0.5 * keyword_match_percentage) + (0.2 * grammar_score) + (0.3 * answer_length_score)
    final_score = (final_score / 100) * total_marks  

    return keyword_matches, total_keywords, keyword_match_percentage, grammar_score, answer_length_score, final_score


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    file = request.files['file']
    total_marks = float(request.form['total_marks'])
    
   
    file_path = os.path.join(UPLOADS_FOLDER, file.filename)
    file.save(file_path)

 
    keyword_matches, total_keywords, keyword_match_percentage, grammar_score, answer_length_score, final_score = evaluate_assignment(file.filename, total_marks)
    
 
    result_path = os.path.join(RESULTS_FOLDER, f"{file.filename}_result.txt")
    with open(result_path, 'w') as f:
        f.write(f"Assignment: {file.filename}\n")
        f.write(f"Keyword Match: {keyword_matches}/{total_keywords} words matched ({keyword_match_percentage:.2f}%)\n")
        f.write(f"Grammar Score: {grammar_score:.2f}%\n")
        f.write(f"Answer Length Score: {answer_length_score:.2f}%\n")
        f.write(f"Final Score: {final_score:.2f}/{total_marks}\n")

   
    return render_template('result.html', 
                           filename=file.filename, 
                           keyword_match_percentage=keyword_match_percentage,
                           grammar_score=grammar_score,
                           answer_length_score=answer_length_score,
                           final_score=final_score)


@app.route('/results/<filename>')
def download_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
