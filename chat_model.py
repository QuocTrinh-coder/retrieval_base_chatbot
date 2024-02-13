#Import all packages we'll need. 
import json
import numpy as np
import random
import nltk
import utils as u
#nltk.download('punkt')
#nltk.download('wordnet')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class ChatModel:
    # cleaning training data for training process
    def __init__(self):
        # Call tokenizing procedure
        w, words, documents, classes, self._intents = self.tokenizing('data.json')

        # Call lemmatizing procedure
        w, words, documents, classes, lemmatizer = self.lemmatizing(w, words, documents, classes)

        # Call training_data procedure
        self._train_x, self._train_y, self._test_x, self._test_y = self.training_data(w, words, documents, classes, lemmatizer)

        # Call tokenizing procedure
        self._model = self.training(self._train_x, self._train_y)

        # Evaluate the model on the test data
        self.evaluate_model(self._test_x, self._test_y)
        
        # confusion matrix, uncomment to see the matrix
        
        #self.print_confusion_matrix(self._test_x, self._test_y)
    # tokenizing method
    def tokenizing(self,url):
        words=[]
        classes = []
        documents = []
        intents = json.loads(open(url).read())

        for intent in intents['intents']:
            for pattern in intent['question']:
                #tokenize each word
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                #add documents in the corpus
                documents.append((w, intent['tag']))
                # add to our classes list
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])

        return w, words, documents, classes, intents

    # lemmatizing method
    def lemmatizing(self, w, words, documents, classes):
        ignore_words = ['?', '!']
        lemmatizer = nltk.stem.WordNetLemmatizer()

        # lemmatize, lower each word and remove duplicates
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

        # sort classes and words
        classes = sorted(list(set(classes)))
        words = sorted(list(set(words)))
        # documents = combination between questions and intents
        print (len(documents), "documents")

        # classes = intents
        print (len(classes), "classes", classes)

        # words = all words, vocabulary
        print (len(words), "unique lemmatized words", words)

        u.create_pickle(words, 'pickles\words.pkl') 
        u.create_pickle(classes, 'pickles\classes.pkl')
        return w, words, documents, classes, lemmatizer

    # obtain training data
    def training_data(self, w, words, documents, classes, lemmatizer, test_size=0.2):
        # create our training data
        training = []
        # create an empty array for our output
        output_empty = [0] * len(classes)

        # training set, bag of words for each sentence
        for doc in documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # lemmatize each word - create base word, in an attempt to represent related words
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
            # create our bag of words array with 1 if the word match is found in the current pattern
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for the current tag (for each pattern)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])

        # Shuffle our features
        random.shuffle(training)

        # Ensure consistent shape by padding shorter sequences
        max_sequence_length = max(len(item[0]) for item in training)
        training = [(item[0] + [0] * (max_sequence_length - len(item[0])), item[1]) for item in training]

        # Split the data into training and testing sets
        train_data, test_data = train_test_split(training, test_size=test_size, random_state=42)

        # Convert training list to NumPy array
        train_data = np.array(train_data, dtype=object)
        test_data = np.array(test_data, dtype=object)

        # Create train and test lists. X - patterns, Y - intents
        train_x = list(train_data[:, 0])
        train_y = list(train_data[:, 1])

        test_x = list(test_data[:, 0])
        test_y = list(test_data[:, 1])

        print("Training and test data created")
        return train_x, train_y, test_x, test_y


    def training(self, train_x, train_y):
        # Sequential from Keras
        # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons, and 3rd output layer contains a number of neurons
        # equal to the number of intents to predict output intent with softmax
        model = Sequential() # class allows you to create a linear stack of layers in your neural network.
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # first layer ( input layer has 128 neurons ) , Specifies the shape of the input data. In this case, it is the number of features in the input data (len(train_x[0]), 
        model.add(Dropout(0.5)) # Dropout is a regularization technique that randomly sets a fraction (here, 0.5 or 50%) of input units to zero during training to prevent overfitting.
        model.add(Dense(64, activation='relu')) # second layer with 64 neurons 
        model.add(Dropout(0.5)) # Dropout is a regularization technique that randomly sets a fraction (here, 0.5 or 50%) of input units to zero during training to prevent overfitting.
        model.add(Dense(len(train_y[0]), activation='softmax')) # The output layer with a number of neurons equal to the number of classes or intents in the training data.

        # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
        sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, momentum=0.9, nesterov=True) # setting the hyper parameter for gradient descent for optimization function below 
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # using SGD ( gradient descent ) as the model optimizer meaning finding the best weight to get the lowest SSE ( or RSS )

        # Fitting and saving the model
        hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1) # fit the model with training x , training y and 200 epochs ( Number of epochs or iterations over the entire training dataset.) , Number of samples per gradient update, Output progress during training.
        model.save('chatbot_model.h5', hist)
        print("Model created")

        return model


    def get_train_x(self):
        return self._train_x

    def get_train_y(self):
        return self._train_y
    
    def get_model(self):
        return self._model

    def get_intents(self):
        return self._intents
    
    # def train_test_split(self, test_size=0.4):
    #     # Split the data into training and testing sets
    #     train_x, test_x, train_y, test_y = train_test_split(self._train_x, self._train_y, test_size=test_size, random_state=40)
    #     return train_x, test_x, train_y, test_y

    def evaluate_model(self, test_x, test_y):
        # Evaluate the model on the test data
        predictions = self._model.predict(np.array(test_x))
        predicted_classes = np.argmax(predictions, axis=1)

        true_classes = np.argmax(test_y, axis=1)

        correct_predictions = np.sum(predicted_classes == true_classes)
        total_samples = len(test_y)

        accuracy = correct_predictions / total_samples
        
        # Calculate mean squared error (MSE)
        mse = np.mean((predicted_classes - true_classes) ** 2)
        
        return mse, accuracy
    def k_fold_cross_validation(self, k=10):
        # Split the data into k folds
        kf = KFold(n_splits=k, shuffle=True)

        # Initialize lists to store MSE and accuracy scores for each fold
        mse_scores = []
        accuracy_scores = []

        for fold, (train_index, test_index) in enumerate(kf.split(self._train_x)):
            print(f"Fold {fold + 1}/{k}:")
            train_x_cv, test_x_cv = np.array(self._train_x)[train_index], np.array(self._train_x)[test_index]
            train_y_cv, test_y_cv = np.array(self._train_y)[train_index], np.array(self._train_y)[test_index]

            # Train the model
            self._model = self.training(train_x_cv, train_y_cv)

            # Evaluate the model
            mse, accuracy = self.evaluate_model(test_x_cv, test_y_cv)
            mse_scores.append(mse)
            accuracy_scores.append(accuracy)

            # Print MSE and accuracy scores for the current fold
            print(f"MSE: {mse}, Accuracy: {accuracy}")

        # Calculate and print average MSE and accuracy scores
        avg_mse = np.mean(mse_scores)
        avg_accuracy = np.mean(accuracy_scores)
        print(f"\nAverage MSE: {avg_mse}, Average Accuracy: {avg_accuracy}")

        return avg_mse, avg_accuracy

    def print_confusion_matrix(self, test_x, test_y):
        # Evaluate the model on the test data
        predictions = self._model.predict(np.array(test_x))
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_y, axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)

        # Print confusion matrix
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.show()

# # Usage
# chat_model = ChatModel()
# train_x, test_x, train_y, test_y = chat_model.train_test_split()

# # Train the model on the training set
# chat_model.training(train_x, train_y)

# # Evaluate the model on the testing set and get accuracy
# test_accuracy = chat_model.evaluate_model(test_x, test_y)
# print("Test Accuracy:", test_accuracy)

# # Assuming you have already created an instance of ChatModel and split the data
# ex = ChatModel()
# avg_mse, avg_accuracy = ex.k_fold_cross_validation(k=5)
# print("Average MSE:", avg_mse)
# print("Average Accuracy:", avg_accuracy)
