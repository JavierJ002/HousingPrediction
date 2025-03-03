import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from scipy import stats

def distributions(df: pd.DataFrame, col: str) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.scatter(data=df, x=col, y='SalePrice', color='red')
    ax1.set_xlabel(col, fontsize=12)
    ax1.set_ylabel('SalePrice', fontsize=12)
    ax1.set_title(f'Scatterplot {col} vs SalePrice', fontsize=14)

    ax2.boxplot(df[col])
    ax2.set_xlabel(col, fontsize=12)
    ax2.set_title(f'Boxplot {col}', fontsize=14)

    sns.histplot(ax=ax3, data=df, x=df[col], color='red')
    ax3.set_title(f'Kernel Density of {col}', fontsize=14)
    ax3.set_xlabel(col, fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)

    if col in df.select_dtypes(include=np.number).columns:
        skew_value = df[col].skew()
        ax3.annotate(f'Skew: {skew_value:.3f}', 
                     xy=(0.95, 0.95), xycoords='axes fraction', 
                     fontsize=12, color='black', ha='right', va='top',
                     bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    plt.show()


def show_categories(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"The var: '{col}' doesnt exist in the dataFrame")
    porcentaje_nulos = df[col].isnull().mean() * 100

    conteo = df[col].value_counts(normalize=True, dropna=False) * 100
    categorias = conteo.index.tolist()
    porcentajes = conteo.values.tolist()
    resultado = pd.DataFrame({
        'Categoría': categorias,
        'Porcentaje (%)': porcentajes
    })
    return resultado

def categorical_anova_tukey(df, target):
    results = []
    
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns

    for cat_var in categorical_vars:
        top_categories = df[cat_var].value_counts().index[:5]
        filtered_df = df[df[cat_var].isin(top_categories)]
        groups = [filtered_df[filtered_df[cat_var] == cat][target] for cat in top_categories]
        if all(len(group) > 1 for group in groups):  
            f_stat, p_value = stats.f_oneway(*groups)
            results.append({'Variable': cat_var, 'F-Statistic': f_stat })
    results_df = pd.DataFrame(results).sort_values(by='F-Statistic', ascending=False)

    return results_df

