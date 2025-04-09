from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
import polars as pl
import requests
from time import perf_counter
from typing import TypeVar
import xmltodict

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


class RiskFreeRates:
    imported_rates: dict[int, dict] = {}
    rate_table: dict[date, defaultdict[int, float]] = defaultdict(dict)

    @classmethod
    def load_raw_risk_free_rates(cls, year: int) -> None:
        data = requests.get(''.join((DataCreationConfig.XLS_RISK_FREE_RATE_URL, str(year))))

        cls.imported_rates[year] = xmltodict.parse(data.text)

    @classmethod
    def create_rate_table(cls, date_var: datetime) -> None:
        if date_var.year not in cls.imported_rates:
            cls.load_raw_risk_free_rates(date_var.year)

        if datetime.date not in cls.rate_table:
            for i in range(len(cls.imported_rates[date_var.year]['pre']['entry'])):
                row = cls.imported_rates[date_var.year]['pre']['entry'][i]['content']['m:properties']

                cls.rate_table[date_var][30] = float(row['d:BC_1MONTH']['#text'])
                cls.rate_table[date_var][60] = float(row['d:BC_2MONTH']['#text'])
                cls.rate_table[date_var][90] = float(row['d:BC_3MONTH']['#text'])
                cls.rate_table[date_var][180] = float(row['d:BC_6MONTH']['#text'])
                cls.rate_table[date_var][360] = float(row['d:BC_1YEAR']['#text'])
                cls.rate_table[date_var][720] = float(row['d:BC_2YEAR']['#text'])
                cls.rate_table[date_var][1080] = float(row['d:BC_3YEAR']['#text'])

    @classmethod
    def calculate_rate(cls, timestamp_date: datetime, expiry_date: datetime) -> float:
        day_range = (expiry_date - timestamp_date).days

        next_smaller = max(filter(lambda x: x < day_range, cls.rate_table[timestamp_date].keys()))
        next_larger = min(filter(lambda x: x > day_range, cls.rate_table[timestamp_date].keys()))

        next_smaller_rate = cls.rate_table[timestamp_date][next_smaller]
        next_lager_rate = cls.rate_table[timestamp_date][next_larger]

        m = (next_lager_rate - next_smaller_rate) / (next_larger - next_smaller)

        rate_at_timestamp = next_smaller_rate + (expiry_date - timestamp_date).days * m

        return round(rate_at_timestamp, 3)

    @classmethod
    def get_rate_at_date(cls, timestamp_date: datetime, expiry_date: datetime) -> float:
        if timestamp_date.date() not in cls.rate_table.keys():
            cls.create_rate_table(timestamp_date)

        return cls.calculate_rate(timestamp_date, expiry_date)


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
        .then(pl.col('identifier').str.extract(r"(C|P)", group_index=0))
        .alias('callput')
    ])
    # TODO: Add risk free rate
    # filled_df = filled_df.with_columns([
    #     pl.col('date').map_elements(RiskFreeRates.get_rate_at_date).alias('risk_free_rate')
    # ])

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