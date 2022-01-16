# Preprocessing

## Available Methods

1) skewcorrect
  
    Plots distplot and probability plot for non-normalized data and after normalizing the provided data.
    Normalizes data using boxcox normalization

    Parameters:

      dataset : pd.DataFrame

          Dataset on which skewness correction has to be done.

      except_columns : list

          Columns for which skewness correction need not to be done.Default = []

    :returns: Scaled Dataset
    :rtype: pd.DataFrame
  
    Example:

        1) All Columns

                    from luciferml.preprocessing import Preprocess as pp

                    import pandas as pd

                    dataset = pd.read_csv('/examples/Social_Network_Ads.csv')
                    prep = pp(dataset, dataset.columns)
                    dataset = prep.skewcorrect(dataset)

        2) Except column/columns

                    from luciferml.preprocessing import Preprocess as pp

                    import pandas as pd

                    dataset = pd.read_csv('/examples/Social_Network_Ads.csv')
                    prep = pp(dataset, dataset.columns, except_columns=['Purchased'])
                    dataset = prep.skewcorrect()
