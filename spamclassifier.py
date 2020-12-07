from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
nltk.download('stopwords')


class SpamClassifier:
    def __init__(self, spam_df):
        self.spam_df = spam_df


class DatPreProcessing(SpamClassifier):
    def __init__(self,spam_df):
        super().__init__(spam_df)

    def process_data(self):
        lemm= WordNetLemmatizer()
        corpus = []
# remove the regular expression
        for i in range(0,len(self.spam_df)):
            data = re.sub('[^a-zA-Z]', ' ', self.spam_df['message'][i])
            data = data.lower()
            data = data.split()
            data = [lemm.lemmatize(word) for word in data if not word in stopwords.words('english')]
            data = ' '.join(data)
            corpus.append(data)
        return corpus

    def tfidf_model(self):
        corpus = self.process_data()
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus).toarray()
# print(vectorizer.get_feature_names())
        y = pd.get_dummies(messages['label'])
        y = y.iloc[:, 1].values
        return X,y

    def train_test_data(self):
        X,y=self.tfidf_model()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        return X_train, X_test, y_train, y_test

    def performe_classification(self,):
        X_train, X_test, y_train, y_test = self.train_test_data()
        spam_detect_model = MultinomialNB().fit(X_train, y_train)
        y_pred = spam_detect_model.predict(X_test)
        print('#####confusion Matrix#######')
        print(confusion_matrix(y_test,y_pred))
        print('#####classification report#####')
        print(classification_report(y_test, y_pred))
        print('#####accuracy######')
        print(accuracy_score(y_pred,y_pred))


if __name__ == "__main__":
    messages = pd.read_csv('SMSSpamCollection', sep='\t',
                           names=["label", "message"])
    obj=DatPreProcessing(messages)
    obj.performe_classification()
