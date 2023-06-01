from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import time
import preprocessor

local_cash = dict()

def fit_and_estimate_score(models, models_cash):
    for name, classifier in models:
        start_time = time.time()
        classifier.fit(X_train, y_train)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cvs = cross_val_score(classifier, X_train, y_train, cv=kfold, scoring='accuracy')

        y_pred = classifier.predict(X_test)
        comp_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred, squared=False)

        local_cash[name] = dict()
        local_cash[name]["accuracy"] = accuracy
        local_cash[name]["cvs"] = cvs.mean()
        local_cash[name]["comp_time"] = comp_time

        print(f"Model: {name}")
        print(f"Accuracy: {accuracy}")
        print(f"CVS: {cvs.mean()}")
        print(f"MSE: {mse}")
        print(f"Computation time: {comp_time}")
        print("\n")
        models_cash[name] = {"accuracy": accuracy, "MSE": mse}


def fit_stacking_with_optimization(base_param_grid, final_param_grid, grid_classifier, name):
    start_time = time.time()
    estimators = []
    for classifier_name, classifier in non_tree_models:
        param_grid = base_param_grid[classifier_name]
        grid_search = GridSearchCV(classifier, param_grid, scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)
        best_classifier = grid_search.best_estimator_
        estimators.append((classifier_name, best_classifier))

    param_grid = final_param_grid
    grid_search = GridSearchCV(grid_classifier, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_final_estimator = grid_search.best_estimator_

    stacked_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=best_final_estimator,
        cv=5)
    stacked_clf.fit(X_train, y_train)
    y_pred = stacked_clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cvs = cross_val_score(stacked_clf, X_train, y_train, cv=kfold, scoring='accuracy')
    end_time = time.time()
    comp_time = end_time - start_time

    cash_name = "S-OPT-" + name
    local_cash[cash_name] = dict()
    local_cash[cash_name]["accuracy"] = stacked_clf.score(X_test, y_test)
    local_cash[cash_name]["cvs"] = cvs.mean()
    local_cash[cash_name]["comp_time"] = comp_time

    print(f"Hyper stacking: {name}")
    print(f"Accuracy: {stacked_clf.score(X_test, y_test)}")
    print(f"CVS: {cvs.mean()}")
    print(f"MSE: {mse}")
    print(f"Computation time: {comp_time}")
    print('\n')


def fit_stacking_model(classifier, name):
    print(f"Stacking for {name}")
    start_time = time.time()
    stacked = StackingClassifier(
        estimators=non_tree_models,
        final_estimator=classifier,
        cv=5
    )
    stacked.fit(X_train, y_train)
    cv_scores = cross_val_score(stacked, X_train, y_train, cv=5)

    end_time = time.time()
    accuracy = stacked.score(X_test, y_test)
    y_pred = stacked.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    comp_time = end_time - start_time

    cash_name = "S-" + name
    local_cash[cash_name] = dict()
    local_cash[cash_name]["accuracy"] = accuracy
    local_cash[cash_name]["cvs"] = cv_scores.mean()
    local_cash[cash_name]["comp_time"] = comp_time

    print(f"Accuracy: {accuracy}")
    print(f"CVS: {cv_scores.mean()}")
    print(f"MSE: {mse}")
    print(f"Computation time: {comp_time}")
    print("\n")


X_train, X_test, y_train, y_test = preprocessor.preprocess_and_split_csv_input('output.csv')

# PART 1 -- Models
tree_models = [('Decision tree', DecisionTreeClassifier()),
               ('Random forest', RandomForestClassifier()),
               ('Gradient boosting', GradientBoostingClassifier())]
non_tree_models = [('Logistic regression', LogisticRegression(C=0.1)),
                   ('SVC', SVC()),
                   ('K-neighbors', KNeighborsClassifier())]

tree_model_cash = dict()
non_tree_model_cash = dict()

print("***************************")
print("Tree-based classifiers:")
print("***************************\n")
fit_and_estimate_score(tree_models, tree_model_cash)
print("***************************")
print("Non tree-based classifiers:")
print("***************************\n")
fit_and_estimate_score(non_tree_models, non_tree_model_cash)

# PART 2 -- Stacking improvement for NON tree based
fit_stacking_model(KNeighborsClassifier(), "K-Neighbors")
fit_stacking_model(LogisticRegression(), "Logistic")
fit_stacking_model(SVC(), "SVC")

# PART 3.1 -- Hyperparameters improvement for LogisticRegression
base_param_grid = {
    'Logistic regression': {'C': [0.1, 1, 10]},
    'SVC': {'C': [0.1, 1, 10]},
    'K-neighbors': {'n_neighbors': [3, 5, 7]}
}
final_param_grid = {'C': [0.1, 1, 10]}
fit_stacking_with_optimization(base_param_grid, final_param_grid, LogisticRegression(), 'Logistic regression')

# PART 3.2 -- Hyperparameters improvement for KNeighborsClassifier
base_param_grid = {
    'Logistic regression': {'C': [0.1, 1, 10]},
    'SVC': {'C': [0.1, 1, 10]},
    'K-neighbors': {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']}
}
final_param_grid = {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']}
fit_stacking_with_optimization(base_param_grid, final_param_grid, KNeighborsClassifier(), 'K-Neighbors')

base_param_grid = {
    'Logistic regression': {'C': [0.1, 1, 10]},
    'SVC': {'C': [0.1, 1, 10]},
    'K-neighbors': {'n_neighbors': [3, 5, 7]}
}
final_param_grid = {'C': [0.1, 1, 10]}
fit_stacking_with_optimization(base_param_grid, final_param_grid, SVC(), 'SVC')

# parameters for graphs
parameters = ['DT', 'RF', 'GB', 'LR', 'SVM', 'KNN', 'Stacked LR', 'Stacked SVM', 'Stacked KNN', 'Opt-Stacked LR',
              'Opt-Stacked SVM', 'Opt-Stacked KNN']

####### Accuracy score graph
accuracy_values = [round(local_cash["Decision tree"]["accuracy"], 3),
                   round(local_cash["Random forest"]["accuracy"], 3),
                   round(local_cash["Gradient boosting"]["accuracy"], 3),
                   round(local_cash["Logistic regression"]["accuracy"], 3),
                   round(local_cash["SVC"]["accuracy"], 3),
                   round(local_cash["K-neighbors"]["accuracy"], 3),
                   round(local_cash["S-K-Neighbors"]["accuracy"], 3),
                   round(local_cash["S-Logistic"]["accuracy"], 3),
                   round(local_cash["S-SVC"]["accuracy"], 3),
                   round(local_cash["S-OPT-Logistic regression"]["accuracy"], 3),
                   round(local_cash["S-OPT-K-Neighbors"]["accuracy"], 3),
                   round(local_cash["S-OPT-SVC"]["accuracy"], 3)]

cvs_values = [round(local_cash["Decision tree"]["cvs"], 3),
                   round(local_cash["Random forest"]["cvs"], 3),
                   round(local_cash["Gradient boosting"]["cvs"], 3),
                   round(local_cash["Logistic regression"]["cvs"], 3),
                   round(local_cash["SVC"]["cvs"], 3),
                   round(local_cash["K-neighbors"]["cvs"], 3),
                   round(local_cash["S-K-Neighbors"]["cvs"], 3),
                   round(local_cash["S-Logistic"]["cvs"], 3),
                   round(local_cash["S-SVC"]["cvs"], 3),
                   round(local_cash["S-OPT-Logistic regression"]["cvs"], 3),
                   round(local_cash["S-OPT-K-Neighbors"]["cvs"], 3),
                   round(local_cash["S-OPT-SVC"]["cvs"], 3)]

x = range(len(parameters))

bar_width = 0.35
fig, ax = plt.subplots(figsize=(10, 10))

acc_bars = ax.bar(x, accuracy_values, bar_width, label='Accuracy')
cvs_bars = ax.bar([i + bar_width for i in x], cvs_values, bar_width, label='CVS')

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Accuracy and cross-validation scores for models')
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(parameters, rotation=45, ha='right')
ax.set_ylim(0.85, max(max(accuracy_values), max(cvs_values)) + 0.01)  # Adjust the offset value (0.1) as needed

for i, knn_bar in enumerate(acc_bars):
    height = knn_bar.get_height()
    ax.text(knn_bar.get_x() + knn_bar.get_width() / 2, height, str(accuracy_values[i]), ha='center', va='bottom')

for i, svn_bar in enumerate(cvs_bars):
    height = svn_bar.get_height()
    ax.text(svn_bar.get_x() + svn_bar.get_width() / 2, height, str(cvs_values[i]), ha='center', va='bottom')

ax.legend()
plt.show()

###### Computation time graph
time_values = [round(local_cash["Decision tree"]["comp_time"], 3),
                   round(local_cash["Random forest"]["comp_time"], 3),
                   round(local_cash["Gradient boosting"]["comp_time"], 3),
                   round(local_cash["Logistic regression"]["comp_time"], 3),
                   round(local_cash["SVC"]["comp_time"], 3),
                   round(local_cash["K-neighbors"]["comp_time"], 3),
                   round(local_cash["S-K-Neighbors"]["comp_time"], 3),
                   round(local_cash["S-Logistic"]["comp_time"], 3),
                   round(local_cash["S-SVC"]["comp_time"], 3),
                   round(local_cash["S-OPT-Logistic regression"]["comp_time"], 3),
                   round(local_cash["S-OPT-K-Neighbors"]["comp_time"], 3),
                   round(local_cash["S-OPT-SVC"]["comp_time"], 3)]

x = range(len(parameters))

bar_width = 0.3

fig, ax = plt.subplots(figsize=(10, 12))

knn_bars = ax.bar(x, time_values, bar_width)

# svn_bars = ax.bar([i + bar_width for i in x], time_values, bar_width, label='SVN')

ax.set_xlabel('Models')
ax.set_ylabel('Time, s')
ax.set_title('Computation time for models')
ax.set_xticks([i + bar_width / 2 for i in x])
ax.set_xticklabels(parameters, rotation=45, ha='right')

for i, knn_bar in enumerate(knn_bars):
    height = knn_bar.get_height()
    ax.text(knn_bar.get_x() + knn_bar.get_width() / 2, height, str(time_values[i]), ha='center', va='bottom')

ax.legend()
plt.show()
