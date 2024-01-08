import json
import random 
from datetime import datetime
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from string import punctuation 
from nltk.corpus import stopwords
stopwords = stopwords.words("english")

import pickle 
SVCModel = pickle.load(open("../chatbot_test/SVCModel.pkl", "rb"))
textVect = pickle.load(open("../chatbot_test/textvec.pkl", "rb"))
tagEnc =  pickle.load(open("../chatbot_test/tagEnc.pkl", "rb"))
responses = json.load(open("../chatbot_test/lookup.json", "r"))

def pre_process(text: str):
    text = " ".join([stemmer.stem(word.lower()) for word in text.split() if word.lower() not in [stopwords, punctuation]])
    text = textVect.transform([text])
    return text

def predict_tag(text_vext):
    result = SVCModel.predict(text_vext)
    result = tagEnc.inverse_transform(result)[0]
    return result

def generate_response(tag:str):
    answer = random.choice(responses[tag])
    return(answer)

# Assuming you have user input stored in a variable called 'user_input'
user_input = "What is first degree burn?"

# Preprocess the user input
processed_input = pre_process(user_input)

# Predict the tag based on the processed input
predicted_tag = predict_tag(processed_input)

# Generate a response based on the predicted tag
response = generate_response(predicted_tag)

# Print or use the response
print(response)
