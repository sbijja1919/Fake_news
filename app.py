from flask import Flask, render_template, request
from run_model import fake_news_det

app = Flask(__name__,template_folder='./templates',static_folder='./static')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True,host='127.0.0.1',port=6200)