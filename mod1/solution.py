import pandas as pd
import logging 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(lineno)s - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# Load data
iris = datasets.load_iris()
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
X = iris_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris.target  # Use the original target format

# Log initial data
iris_data['target'] = iris.target
logger.debug(iris_data.head())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Initialize and fit KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Plot data (optional)
# target_names = iris.target_names
# _, ax = plt.subplots()
# scatter = ax.scatter(iris_data['sepal length (cm)'], iris_data['sepal width (cm)'], c=iris_data['target'])
# ax.set_xlabel('Sepal Width (cm)')
# ax.set_ylabel('Sepal Length (cm)')
# ax.set_title('Original Iris Data Set')
# ax.legend(handles=scatter.legend_elements()[0], labels=target_names, loc="lower right", title="Species")
# plt.show()

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

logger.debug(f'Confusion Matrix:\n{conf_matrix}')
logger.debug(f'Classification Report:\n{class_report}')

# Print the evaluation results
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
