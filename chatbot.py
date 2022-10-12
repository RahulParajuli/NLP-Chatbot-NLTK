import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('./data/intents.json').read())

words = pickle.load(open('./models/words.pkl', 'rb'))
classes = pickle.load(open('./models/classes.pkl', 'rb'))
model = load_model('./models/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        print(return_list)
    return return_list

predict_class("what is your name?", model)
predict_class ("what is the time?", model)
predict_class ("When are you open?", model)
predict_class ("thats helpful.", model)
predict_class ("How do i Pay with esewa?", model)
predict_class ("what are the mode of payment available?", model)
predict_class ("can you guys deliver?", model)
predict_class ("Tell me a joke", model)
predict_class ("what is your name?", model)
predict_class ("what is your age", model)
predict_class ("How old are you?", model)
predict_class ("what is your restaurant called?", model)
predict_class ("what is your restaurant name?", model)
predict_class ("what is on your menu?", model)
predict_class ("what is your phonenumber?", model)
predict_class ("what is your email?", model)
