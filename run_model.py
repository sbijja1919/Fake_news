import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


lemmatizer = WordNetLemmatizer()
stpwrds = list(stopwords.words('english'))

model_file = './detect_fake_news.pkl'
tfidf_file = './tfidf.pkl'

# load the model from disk
loaded_model = pickle.load(open(model_file, 'rb'))
tfidf_model = pickle.load(open(tfidf_file, 'rb'))

def fake_news_det(news):
    corpus = []
    if len(news) == 0:
        return "Please provide News input."
    else:
        review = news
        review = re.sub(r'[^a-zA-Z\s]', '', review)
        review = review.lower()
        review = nltk.word_tokenize(review)
        for y in review :
            if y not in stpwrds :
                corpus.append(lemmatizer.lemmatize(y))     
        input_data = [' '.join(corpus)]
        vectorized_input_data = tfidf_model.transform(input_data)
        prediction = loaded_model.predict(vectorized_input_data)
        if prediction[0] == 0:
            return "Prediction of the News :  Looking Fake⚠ News📰 "
        else:
            return "Prediction of the News : Looking Real News📰 "