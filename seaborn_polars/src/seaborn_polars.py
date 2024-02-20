import gc
from copy import deepcopy
from functools import singledispatch
from typing import Any, Callable, Union

import polars as pl
import seaborn as sns
import pandas as pd
from numpy.typing import NDArray


def seaborn_polars(func: Callable[[Any], Any]):
    
    @singledispatch
    def wrapper(data: Union[pd.DataFrame, NDArray[Any]], *args, **kwargs):
        func(data, *args, **kwargs)
        
    @wrapper.register
    def _(data: pl.DataFrame, *args, **kwargs):
        temp = deepcopy(data).to_pandas()
        func(temp, *args, **kwargs)
    
    @wrapper.register
    def _(data: pl.LazyFrame, *args, **kwargs):
        temp = deepcopy(data).collect().to_pandas()
        func(temp, *args, **kwargs)
        del(temp)
        gc.collect()
        
    return wrapper


@seaborn_polars
def relplot(data, *args, **kwargs):
    return sns.relplot(data, *args, **kwargs)

@seaborn_polars
def lineplot(data, *args, **kwargs):
    return sns.lineplot(data, *args, **kwargs)    

@seaborn_polars
def scatterplot(data, *args, **kwargs):
    return sns.scatterplot(data, *args, **kwargs)
    
@seaborn_polars
def displot(data, *args, **kwargs):
    return sns.displot(data, *args, **kwargs)
    
@seaborn_polars
def histplot(data, *args, **kwargs):
    return sns.histplot(data, *args, **kwargs)
    
@seaborn_polars
def kdeplot(data, *args, **kwargs):
    return sns.kdeplot(data, *args, **kwargs)
    
@seaborn_polars
def ecdfplot(data, *args, **kwargs):
    return sns.ecdfplot(data, *args, **kwargs)
    
@seaborn_polars
def rugplot(data, *args, **kwargs):
    return sns.rugplot(data, *args, **kwargs)
    
@seaborn_polars
def catplot(data, *args, **kwargs):
    return sns.catplot(data, *args, **kwargs)
    
@seaborn_polars
def stripplot(data, *args, **kwargs):
    return sns.stripplot(data, *args, **kwargs)
    
@seaborn_polars
def swarmplot(data, *args, **kwargs):
    return sns.swarmplot(data, *args, **kwargs)
    
@seaborn_polars
def boxplot(data, *args, **kwargs):
    return sns.boxplot(data, *args, **kwargs)
    
@seaborn_polars
def violinplot(data, *args, **kwargs):
    return sns.violinplot(data, *args, **kwargs)
    
@seaborn_polars
def boxenplot(data, *args, **kwargs):
    return sns.boxenplot(data, *args, **kwargs)
    
@seaborn_polars
def pointplot(data, *args, **kwargs):
    return sns.pointplot(data, *args, **kwargs)
    
@seaborn_polars
def barplot(data, *args, **kwargs):
    return sns.barplot(data, *args, **kwargs)
    
@seaborn_polars
def countplot(data, *args, **kwargs):
    return sns.countplot(data, *args, **kwargs)
    
@seaborn_polars
def lmplot(data, *args, **kwargs):
    return sns.lmplot(data, *args, **kwargs)
    
@seaborn_polars
def regplot(data, *args, **kwargs):
    return sns.regplot(data, *args, **kwargs)
    
@seaborn_polars
def residplot(data, *args, **kwargs):
    return sns.residplot(data, *args, **kwargs)
    
@seaborn_polars
def heatmap(data, *args, **kwargs):
    return sns.heatmap(data, *args, **kwargs)
    
@seaborn_polars
def clustermap(data, *args, **kwargs):
    return sns.clustermap(data, *args, **kwargs)
    
@seaborn_polars
def pairplot(data, *args, **kwargs):
    return sns.pairplot(data, *args, **kwargs)
    
@seaborn_polars
def jointplot(data, *args, **kwargs):
    return sns.jointplot(data, *args, **kwargs)
