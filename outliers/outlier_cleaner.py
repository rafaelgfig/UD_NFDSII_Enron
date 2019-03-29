#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import pandas as pd
    df = pd.DataFrame(net_worths, columns=['net'])
    df['predict'] = predictions
    df['error'] = (df['predict'] - df['net'])**2
    df['ages'] = ages
    df = df.sort_values('error',ascending=False).tail(-int(df.shape[0]*0.1))
    cleaned_data = df[['ages','net', 'error']].sort_index().values.tolist()
    
    return cleaned_data

