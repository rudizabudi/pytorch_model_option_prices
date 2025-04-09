from dataclasses import dataclass
import polars as pl
from time import perf_counter
from typing import TypeVar

from config import DataCreationConfig

OptionTableType = TypeVar('OptionTableType')
StockTableType = TypeVar('StockTableType')

@dataclass
class OptionData:
    identifier: str
    data: list[OptionTableType]

@dataclass
class StockData:
    identifier: str
    data: list[StockTableType]

def opt_create_proto_df(option_data: OptionData) -> pl.DataFrame:
    data_list = []
    for row in option_data.data:
        data_list.append({
            'date': row.date,
            'identifier': row.identifier,
            'callput': row.callput,
            'strike': row.strike,
            'h': row.h,
            'l': row.l,
            'o': row.o,
            'c': row.c})
    
    return pl.DataFrame(data_list)

def opt_fill_missing_rows(df: pl.DataFrame) -> pl.DataFrame:

    dates = df.select('date').unique()
    identifiers = df.select('identifier').unique()

    full_index = dates.join(identifiers, how='cross')

    filled_df = full_index.join(df, on=['date', 'identifier'], how='left')

    filled_df = filled_df.with_columns([
        pl.when(pl.col('strike').is_null())
        .then(pl.col('identifier').str.extract(r"(\d+\.\d+)", group_index=0).cast(pl.Float64))
        .alias('strike'),

        pl.when(pl.col('callput').is_null())
        .then(pl.col('identifier').str.extract(r"_(C|P)_", group_index=0))
        .alias('callput')
    ])

    return filled_df

def opt_underlying_merge(df: pl.DataFrame, underlying_prices: pl.DataFrame):
    underlying_prices.rename({'h': 'ul_h', 'l': 'ul_l', 'o': 'ul_o', 'c': 'ul_c'})

    df = df.join(underlying_prices, on=['date'], how='left')
    return df

def stk_create_df(stock_data: StockData) -> pl.DataFrame:
    data_list = []
    for row in stock_data.data:
        data_list.append({
            'date': row.date,
            'h': row.h,
            'l': row.l,
            'o': row.o,
            'c': row.c})
    
    return pl.DataFrame(data_list)


def option_data_to_polars(option_data: OptionData, underlying_prices: pl.DataFrame):
    proto_df = opt_create_proto_df(option_data)

    filled_df = opt_fill_missing_rows(proto_df)

    merged_df = opt_underlying_merge(filled_df, underlying_prices)

    return merged_df

def stk_data_to_polars(stock_data: StockData):
    df = stk_create_df(stock_data)

    return df