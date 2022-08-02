# IMPORTAMOS LAS LIBRERÍAS NECESARIAS
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay

# Elegimos el numero de vecinos que queremos usar para la classificación
n_neighbors = 15

# Usaremos el dataset iris
iris = datasets.load_iris()

# Seleccionamos solo dos features
# porque haremos una grafica en dos dimensiones
# si usamos un dataset de dos dimensiones esto no es necesario
X = iris.data[:, :2]
print('---------- Esta es la X ----------------')
print(X)
y = iris.target

# creamos los mapas de colores
# Puedes elegir otros colores por supuesto
cmap_light = ListedColormap(["gray", "cyan", "lightblue"])
cmap_bold = ["darkorange", "red", "darkblue"]


# Entrenamos el MODELO
# Para ver la documentación de neighbours.KNeighborsClassifies puedes ir a: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# Ahí puedes ver el uso de weights
# uniform le da a cada vecino el mismo peso
# distance da mas peso a los vecinos mas cercanos (dentro del numero de vecinos obvio).
for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

#   Aqui hacemos la grafica 
#   Extraemos de plt.subplots() dos variables solo nos interesa ax
#   plt.subplots() retorna: fig, ax = plt.subplots()
    variable_que_no_usamos, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
        shading="auto",
    )

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=iris.target_names[y],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="brown",
    )
    plt.title(
        "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
    )

plt.show()


#----------------------------------------------------------
#----------------------------------------------------------
#----------------------------------------------------------

#--------------------------------------------------------------------------
# Puedes utilizar el siguiente código para responder la pregunta de la tarea  
#--------------------------------------------------------------------------
from sklearn.datasets import load_iris
iris = load_iris()

print(iris.target_names)
print(iris.data.shape)

X = iris.data[:, :4]
y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

range_k = range(1,n_neighbors)
scores = {}
scores_list = []
for k in range_k:
   classifier = KNeighborsClassifier(n_neighbors=k)
   classifier.fit(X_train, y_train)
   y_pred = classifier.predict(X_test)
   scores[k] = metrics.accuracy_score(y_test,y_pred)
   scores_list.append(metrics.accuracy_score(y_test,y_pred))
print("Lista de scores: ", scores_list)
result = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
metricas = metrics.classification_report(y_test, y_pred)
print("Classification Report:",)
print (metricas)

