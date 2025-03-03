import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from scipy import stats

def make_submission(preds, name):
    predictions = pd.DataFrame()
    name = name.replace(' ', '')
    og_test_set = pd.read_csv('..//data//raw_data//test.csv')
    predictions['Id'] = og_test_set['Id']
    predictions['SalePrice'] = preds
    predictions = predictions[['Id', 'SalePrice']]
    predictions.to_csv(f'..//data//predictions//{name.lower()}.csv', index = False)
    return predictions

def plot_residuals_analysis(models_dict, X, y):

    num_models = len(models_dict)
    fig, axes = plt.subplots(num_models, 2, figsize=(15, 5 * num_models))

    for idx, (model_name, model) in enumerate(models_dict.items()):
        try:

            y_pred = cross_val_predict(model, X, y, cv=10)
            residuals = y - y_pred


            ax1 = axes[idx, 0] if num_models > 1 else axes[0]
            sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax1)
            ax1.axhline(0, color='red', linestyle='--')
            ax1.set_title(f'{model_name.upper()} - Residuals vs Predictions')
            ax1.set_xlabel('Predictions')
            ax1.set_ylabel('Residuals')

            # QQ-Plot
            ax2 = axes[idx, 1] if num_models > 1 else axes[1]
            stats.probplot(residuals, dist='norm', plot=ax2)
            ax2.set_title(f'{model_name.upper()} - QQ-Plot de Residuals')

        except Exception as e:
            print(f'Errors of the model {model_name}: {str(e)}')

    plt.tight_layout()
    plt.show()