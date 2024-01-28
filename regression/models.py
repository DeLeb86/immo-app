from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV

def get_model():
    return GradientBoostingRegressor(**{
        'n_estimators': 800,
        'learning_rate':  0.05,
        'subsample':0.7,
        'max_depth': 9
        })

# evaluate a model
def evaluate_model(X, y, model):
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1,verbose=2)
    scores = absolute(scores)
    return scores

def plot_perf(models,results):
    for name, model in models.items():
        print('>%s %.3f (%.3f)' % (name, mean(results[name]), std(results[name])))
    #plot model performance for comparison
    pyplot.boxplot(results.values(), labels=results.keys(), showmeans=True)
    pyplot.legend()
    pyplot.show()

def grid_search_hyper(model,param_grid,X_train,y_train):
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid,scoring="r2",
                        cv = 3, n_jobs = -1, verbose = 3)
    grid_search.fit(X_train,y_train)
    return grid_search.best_params_, grid_search.best_estimator_
    