# Here, SVM is selected as the classifier class to classify and predict.
"""PCANet Class"""
from sklearn import svm
import time

class ClassifierTool:
    def __init__(self,linear_classifier='svm'):
        #Whether to use linear classification SVM code
        if linear_classifier == 'svm':
            self.classifier = svm.SVC(probability=True)
        else:
            self.classifier = None

    #Set the saved classifier after training
    def setClassifier(self, classifier):
        self.classifier = classifier

    #Acquire trained classifiers
    def getClassifier(self):
        return self.classifier

    #Enter the features and the training tag, so that you can
    def trainSvm(self,features,train_labels):
        print('features extracted, SVM training')
        self.classifier.fit(features, train_labels)

        #Save Model
        # joblib.dump(self.classifier, '../model/classifiersvm.pkl')
        # print(self.classifier.get_params())

    #Obtain accurate labels for prediction based on input characteristics
    def SVM_predict(self,predictionFeature):
        self.classifier.predict(predictionFeature)

    #The prediction probability of each tag predicted is obtained according to the input characteristics
    def SVM_predict_proba(self,predictionFeature):
        self.classifier.predict_proba(predictionFeature)

    # It is mainly improved to speed up the whole code
    # Obtain accurate labels for prediction based on input characteristics
    def SVM_predictFast(self, predictionFeature,predictionTotal = 0):
        # If the total forecast quantity is not entered, it will be obtained by default
        # Data of the second dimension of predictionFeature
        if predictionTotal == 0:
            predictionTotal = predictionFeature.shape[1]

        # Modify the logic to separate each prediction
        # Predictions
        predictions = []
        index = 0

        # every_ PredicationSize The size of each prediction data interval. Here, select 500 features to put into the data set
        # Some improvements can be made. See the input parameters for details
        # test_ Features are ready to predict the feature information
        every_PredictionSize = 500
        # When forecasting
        if(predictionTotal < every_PredictionSize):
            every_PredictionSize = predictionTotal
        start = time.time()

        #The idea here is test_ Features will also be put into the classifier for classification to get the final result
        for i in range(every_PredictionSize,predictionTotal,every_PredictionSize):
            test_features = [predictionFeature[i-every_PredictionSize:i]]
            index = index + every_PredictionSize
            print('predicting', i, 'th label')
            # Since there is no svm. SVC (probability=True), the probability point cannot be obtained
            # Keep every result
            predictions.append(self.SVM_predict(test_features))

        #Predict the rest of the final predicted value
        if(index<predictionTotal):
            predictions.append(self.SVM_predict([predictionFeature[index:predictionTotal]]))

        end = time.time()
        print('It takes time to forecast multiple data :', end - start)
        print('=' * 20)
        return predictions



