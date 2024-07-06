import pandas as pd
import logging 

import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(lineno)s: \n%(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

def remove_outliers(col: pd.Series):
    logger.info(f'Removing outliers for  {col.name}')
    q1, q3= col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr
    
    outlier_mask = [(col>lower_bound) & (col<upper_bound)]
    logger.info(f'Identified {len(outlier_mask)} outliers.')
    
    col[outlier_mask] = pd.NA
    return col

# Load data
iris = datasets.load_iris()
logger.debug('Iris dataset imported successfully.')
iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_data['target'] = iris.target

# Plot data
target_names = iris.target_names
_, ax = plt.subplots()
scatter = ax.scatter(y=iris_data['sepal length (cm)'], x=iris_data['sepal width (cm)'],c=iris_data['target'])
ax.set_xlabel('Sepal Width (cm)')
ax.set_ylabel('Sepal Length (cm)')
ax.set_title('Original Iris Data Set')
ax.legend(scatter.legend_elements()[0], target_names, loc="lower right", title="Species")
plt.show()

pd.set_option('display.float_format',lambda x: '%.3f' % x)
logger.info(f'Iris Dataset Description \n:{iris_data.describe()}')

logger.info(f' Number of Null values observed: {iris_data.isnull().sum()}')

for col in iris_data.columns:
    q1 = iris_data[col].quantile(0.25)
    q3 = iris_data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    iris_data = iris_data[(iris_data[col]>lower_bound) & (iris_data[col]<upper_bound)]
logger.info(f'Len is {len(iris_data)}')


X = iris_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = iris_data['target']

_, ax = plt.subplots()
hist = ax.boxplot(X)
ax.set_xticklabels(['Sepal Length', 'Sepal Width','Petal Length', 'Petal Width'])
ax.set_title('Distribution of Iris Dataset')
plt.show()

logger.debug((iris_data.head()))
# Assign train and test datasets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
logger.info(f'{len(X_train)} observations to train model.\n{len(X_test)} observations to test model.')
# Configure and fit the model.
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

logger.info('Confusion Matrix')
logger.info(confusion_matrix(y_test, y_pred))
logger.info('Classification Report')
logger.info(classification_report(y_test, y_pred, target_names=iris.target_names))
