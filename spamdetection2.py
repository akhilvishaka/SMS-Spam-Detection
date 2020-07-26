import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


dataSet = pd.read_csv('spam.csv', encoding='cp437')
y = dataSet.iloc[:, 0].values
stemmedReviews = []


for i in range(y.size):
    review = re.sub('[^a-zA-Z]', ' ', dataSet['v2'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    stemmedReviews.append(review)

cv = CountVectorizer(max_features=3000)
x = cv.fit_transform(stemmedReviews).toarray()

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)

print("\nNAIVE BAYES : \n")
print('Accuracy score: {}'.format(accuracy_score(y_test, pred)*100))
print('Precision score: {}'.format(precision_score(y_test, pred)*100))
print('Recall score: {}'.format(recall_score(y_test, pred)*100))
print('F1 score: {}'.format(f1_score(y_test, pred)*100))

classifier1 = RandomForestClassifier(n_estimators=15, criterion='entropy')
classifier1.fit(X_train, y_train)
predRF = classifier1.predict(X_test)
print("\nRANDOM FOREST CLASSIFIER : \n")
print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)*100))
print('Precision score: {}'.format(precision_score(y_test, predRF)*100))
print('Recall score: {}'.format(recall_score(y_test, predRF)*100))
print('F1 score: {}'.format(f1_score(y_test, predRF)*100))

stemmedReviews = []

dataset = pd.read_csv('test1.csv', encoding='cp437')

for i in range(0, 1):
    review = re.sub('[^a-zA-Z]', ' ', dataset['v1'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    stemmedReviews.append(review)

temp = cv.transform(stemmedReviews)
predNaiveBayes = classifier.predict(temp.toarray())
predRandomForest = classifier1.predict(temp.toarray())

if predNaiveBayes == 1:
    OutputNB = "Spam"

else:
    print("Not spam")
    OutputNB = "Not spam"

if predRandomForest == 1:
    print("Spam")
    OutputRF = "Spam"
else:
    print("Not spam")
    OutputRF = "Not spam"

print("\nOutput for Gaussian NB = {0} {1}".format(classifier.predict(temp.toarray()), OutputNB))
print("\nOutput for Random Forest Classifier = {0} {1}".format(classifier1.predict(temp.toarray()), OutputRF))
