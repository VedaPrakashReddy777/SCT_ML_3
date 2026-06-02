from PIL import Image
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = []
labels = []

for file in os.listdir("dataset/cats"):
    img = Image.open(os.path.join("dataset/cats", file))
    img = img.resize((64, 64))
    img = np.array(img).flatten()
    data.append(img)
    labels.append(0)

for file in os.listdir("dataset/dogs"):
    img = Image.open(os.path.join("dataset/dogs", file))
    img = img.resize((64, 64))
    img = np.array(img).flatten()
    data.append(img)
    labels.append(1)

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("SVM Model Accuracy:", accuracy * 100, "%")