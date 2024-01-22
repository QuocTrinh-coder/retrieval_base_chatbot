# Data Processing libraies
import json
import pandas as pd
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
stopwords = stopwords.words("english")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#visualization libraries
import seaborn as sns 
from matplotlib import pyplot as plt

# ML : model training libraries
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier

# ML : model eval 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# object serializer 
import pickle 

# Step 1: Read in the dataset
with open('data.json', 'r') as file:
    data = json.load(file)

# Step 2: Store the dataset in a variable
intents_data = data['intents']

# Step 3: Check the data description
# print("Data Description:")
# print("Number of intents:", len(intents_data))
# print()

# Step 4: Convert the data to a pandas DataFrame
dataFrame = pd.DataFrame(intents_data)

# Step 5: Convert categorical data to string type (if necessary)
dataFrame['question'] = dataFrame['question'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

# Display the DataFrame
# print("DataFrame:")
# print(dataFrame)

# Display information about the DataFrame
# print("\nDataFrame Information:")
# print(dataFrame.info())

# Instantiating objects
stemmer = PorterStemmer()
label_encoder = LabelEncoder()
label_encoder.fit(dataFrame["tag"])
# Saving Encoder
with open("tagEnc.pkl", "bw") as file:
    pickle.dump(label_encoder, file)
# reading in saving encoder
label_encoder = pickle.load(open("tagEnc.pkl", "rb"))

# transforming data and updating dataframe
dataFrame["clean_questions"] = dataFrame["question"].apply(lambda x : " ".join([stemmer.stem(word.lower()) for word in x.split()
                                                                             if word.lower() not in [stopwords, punctuation]]))
dataFrame["tag_labels"] = dataFrame["tag"].apply(lambda x : label_encoder.transform([x])).apply(lambda x: x[0])
dataFrame = dataFrame.sample(frac = 1, ignore_index = True) # shuffle and sample the dataset with 100% data

#print(dataFrame.head().to_string())

# Instantiating objects
# ngram_range = (1,3) was chosen after experiments on its effect on accuracy
textVect = TfidfVectorizer(stop_words="english" , ngram_range=(1,3))

# fitting text to vectorizers
textVect.fit([word for word in dataFrame["clean_questions"]])

# saving vectorizers
with open("textvec.pkl", "bw") as file:
        pickle.dump(textVect,file)
        
# reading saved vectorizers
textVect = pickle.load(open("textVec.pkl", "br"))

# Splitting data
x = dataFrame["clean_questions"]
y = dataFrame["tag_labels"]

x = textVect.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state=32)

print("X training data shape" , x_train.shape)
print("X test data shape" , x_test.shape)

SVCModel = SVC()
SVCModel.fit(x_train,y_train)

# saving model
with open("SVCModel.pkl", "bw") as file:
        pickle.dump(SVCModel,file)
        
# model Evaluation 
pred = SVCModel.predict(x_test)
score = accuracy_score(y_test, pred)

tpred = SVCModel.predict(x_train)
tscore = accuracy_score(y_train, tpred)

print("test accuracy", score)
print("train accuracy", tscore)

# Assuming pred and y_test are the predicted and true labels
conf_matrix = confusion_matrix(y_test, pred)

# # Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()