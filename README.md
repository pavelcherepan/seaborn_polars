# seaborn_polars
Wrapper allowing to use Polars DataFrames and LazyFrames for plotting with seaborn

# Requirements
Python 3.8+

Seaborn, Polars, Pandas, Numpy, Pyarrow

# Installation
Use regular pip install:
```
pip install seaborn_polars pandas numpy polars seaborn pyarrow
```

Install from the main branch on GitHub:
```
pip install pip install "git+https://github.com/pavelcherepan/seaborn_polars#egg=seaborn_polars"
```

You may need to run `pip install --upgrade pip` first.

Alternatively, clone this repository and run `pip install .` from the project root directory.

# Usage
The package is a wrapper around seaborn plotting functions allowing to use Polars DataFrames and LazyFrames with the same syntax as when using Pandas DataFrames.
```
import polars as pl
import seaborn_polars as snl

df = pl.scan_csv('data.csv')

snl.scatterplot(df, x='rating', y='responses', hue='gender')
```
The package uses a deepcopy of original data. That way the original dataframe remains unaffected. For example, in the code snippet above, if we check the type of `df` after plotting we'll see that it is still a Polars LazyFrame.

All the plotting functions and parameters are identical to seaborn. When using data parameter, it has to be positional-only, the rest of parameters are keyword arguments.