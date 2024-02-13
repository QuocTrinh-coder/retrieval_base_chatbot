from chat_model import ChatModel as chatModel
import nltk
import pickle
import numpy as np
from keras.models import load_model #  A function from Keras to load a pre-trained neural network model.
import json
import random
import utils as u
import os

class ChatApp:
    def __init__(self, model, lemmatizer, words, classes, intents):
        self._model = model
        self._lemmatizer = lemmatizer
        self._words = words
        self._classes = classes
        self._intents = intents

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self._lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words) 
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        ERROR_THRESHOLD = 0.25
        p = self.bow(sentence, self._words, show_details=False)
        res = self._model.predict(np.array([p]))[0]

        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self._classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self._intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.predict_class(text)
        res = self.getResponse(ints)
        return res

# # Load pre-trained model and necessary data
# pre_trained_model = load_model('chatbot_model.h5')
# lemmatizer = nltk.stem.WordNetLemmatizer()
# words = u.load_pickle(os.path.join('pickles', 'words.pkl'))
# classes = u.load_pickle(os.path.join('pickles', 'classes.pkl'))
# intents = json.loads(open('data.json').read())

# # Create ChatApp instance with pre-trained model and data
# ex = ChatApp(pre_trained_model, lemmatizer, words, classes, intents)

# # Test the chatbot response
# print(ex.chatbot_response("can you tell me what is first degree burn ?"))
# print(ex.chatbot_response("can you tell me what is second degree burn ?"))