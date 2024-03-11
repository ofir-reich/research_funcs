"""Functions for statistics and tabular data analysis."""

import pandas as pd
import re

def is_fe_var(variable_name):
    """Whether the variable name indicates a Fixed Effect, C(<something>)."""
    return bool(re.match(r'C\(.*\)', variable_name))


def has_fe(model):
    """Whether the model has Fixed Effects."""
    exog_names = model.exog_names
    if any([is_fe_var(exog_name) for exog_name in exog_names]):
        return 'Yes'
    else:
        return 'No'


def reg_summary(fitted_model, exclude_fe=True):
    """Regression summary as a DataFrame, not the statsmodels returned object.
    
    :param fitted_model: statsmodels regression model, with .fit()
    :param exclude_fe: Whether to exclude Fixed Effects variables in the returned coefficients table.

    :return: DataFrame with coefficients, standard errors, p-values, CIs etc.
    """
    sm_summary = fitted_model.summary()
    summary_df = pd.DataFrame(sm_summary.tables[1][1:], columns=sm_summary.tables[1][0])
    summary_df.columns = summary_df.columns.astype(str)
    summary_df.rename(columns={'': 'variable'}, inplace=True)
    # Change data types.
    summary_df['variable'] = summary_df['variable'].astype(str)
    for col_name in summary_df.columns[1:]:
        summary_df[col_name] = summary_df[col_name].astype(str).astype(float)
    if exclude_fe:
        summary_df = summary_df[~summary_df['variable'].apply(is_fe_var)]
        
    return summary_df
