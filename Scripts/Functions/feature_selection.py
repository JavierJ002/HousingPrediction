from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold

def sfs(model, X_train, y_train, k_f = 'parsimonious', method_selection = False, metric = 'neg_mean_squared_error', plot = True):
    sfs = SFS(model, k_features = k_f, forward= method_selection, cv=KFold(n_splits=5, shuffle=True, random_state=2025),
                scoring = metric, n_jobs= -1)
    sfs.fit(X_train, y_train)
    results = pd.DataFrame(sfs.get_metric_dict()).T
    cvs = 5

    if plot == True:
        results['std_dev'] = results['cv_scores'].apply(lambda x: np.std(x))
        results['stderr'] = results['std_dev'] / np.sqrt(cvs)
        results['num_features'] = results.index.astype(int)
        min_score_idx = abs(results['avg_score']).idxmin()
        min_score_mse = -results.loc[min_score_idx, 'avg_score']
        min_score_features = results.loc[min_score_idx, 'num_features']

        plt.figure(figsize=(12, 5))
        plt.errorbar(results['num_features'], -results['avg_score'], yerr=results['stderr'],
                    fmt='o-', color='blue', ecolor='red', capsize=5)

        plt.scatter(min_score_features, min_score_mse, marker='o', color='red',
                    edgecolors='black', s=150, label=f'Optimal: {min_score_features} features')

        plt.xlabel('Number of selected features', fontsize=10)
        plt.ylabel('MSE', fontsize=12)
        plt.ylim(min_score_mse - 0.0025, min_score_mse + 0.0025)
        plt.title('Feature selection with SFS', fontsize=14)
        plt.grid(alpha=0.5)
        plt.legend()
        plt.show()

    return results, sfs

#model40 = list(results.loc[results.index == 48]['feature_names'])
#model40
def get_selected_vars(sfs, info = True):
    vars = list(sfs.k_feature_names_)
    score = sfs.k_score_
    if info == True:
        print(f'The number of variables selected was: {len(vars)}')
        for i, var in enumerate(vars):
            print(f'{i} - {var}')
        print(f'The mse obtained with cv=5 was: {-score}')
    return list(sfs.k_feature_names_)

def cross_val_mse_score(model, X_train, y_train, info = False):
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error',
                            cv=KFold(n_splits=5, shuffle=True, random_state=2025), n_jobs=-1)
    mse = -np.mean(scores)
    std = np.std(scores)


    if info:
        print(f'The MSE obtained was: {mse}')
        print(f'The std obtained was: {std}')

    return mse

def lineplot(score_dict,  title_ = '', xlab = ''):
    depths = score_dict.keys()
    min_score_depth =  min(score_dict, key= score_dict.get)
    min_score = score_dict[min_score_depth]

    plt.figure(figsize = (12, 6))
    plt.plot(depths, score_dict.values(), marker = 'o', color = 'blue')
    plt.plot(min_score_depth, min_score, marker = 'o', color = 'red', label = 'lowest MSE')
    plt.title(title_)
    plt.xlabel(xlab)
    plt.ylabel('MSE')
    plt.grid(alpha = 0.6)
    plt.legend()
    plt.show()