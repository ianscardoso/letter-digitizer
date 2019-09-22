from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
import script as script

# This creates our first MLP with 1 hidden layer with 50 neurons and sets it to run through the data 20 times
mlp1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

print("Created our first MLP network")

mlp1.fit(script.X_train, script.y_train)
print("Training set score: %f" % mlp1.score(script.X_train, script.y_train))
print("Test set score: %f" % mlp1.score(script.X_test, script.y_test))