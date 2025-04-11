from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import polars as pl
import shutil
from sqlalchemy import create_engine, String, Float, text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker
from sqlalchemy.exc import InvalidRequestError, SAWarning
from sqlalchemy.ext.declarative import declarative_base
import subprocess
import tempfile
from time import perf_counter_ns, sleep
from typing import TypeVar
import warnings

from convert_to_df import option_data_to_polars, stk_data_to_polars
from config import DataCreationConfig

if os.getenv('DEV_VAR') == 'rudizabudi':
    DEV_MODE = True
    DataCreationConfig.SQL_CREDENTIALS_JSON_PATH = 'sql_credentials_dev.json'

    LOGGING = False
    if LOGGING:
        import logging

        logging.basicConfig()
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

warnings.filterwarnings('ignore', category=SAWarning)

OptionTableType = TypeVar('OptionTableType')
StockTableType = TypeVar('StockTableType')

Base = declarative_base()


def build_OptionTable(table_name):
    class OptionTable(Base):
        __tablename__ = table_name

        date: Mapped[datetime] = mapped_column(primary_key=True)
        identifier: Mapped[str] = mapped_column(String(50), primary_key=True)
        callput: Mapped[str] = mapped_column(String(1))
        strike: Mapped[float] = mapped_column(Float)
        h: Mapped[float] = mapped_column(Float)
        l: Mapped[float] = mapped_column(Float)
        o: Mapped[float] = mapped_column(Float)
        c: Mapped[float] = mapped_column(Float)

        def __repr__(self):
            return (f"<OptionTable(date={self.date}, identifier='{self.identifier}', "
                    f"callput='{self.callput}', strike={self.strike}, h={self.h}, "
                    f"l={self.l}, o={self.o}, c={self.c})>")
    
    return OptionTable


def build_StockTable(table_name: str) -> StockTableType:
    class StockTable(Base):
        __tablename__ = table_name

        date: Mapped[datetime] = mapped_column(primary_key=True)
        h: Mapped[float] = mapped_column(Float)
        l: Mapped[float] = mapped_column(Float)
        o: Mapped[float] = mapped_column(Float)
        c: Mapped[float] = mapped_column(Float)

        def __repr__(self):
            return (f'<StockTable(date={self.date}, h={self.h}, '
                    f'l={self.l}, o={self.o}, c={self.c})>')
    
    return StockTable

    
@dataclass
class OptionData:
    identifier: str
    data: list[OptionTableType]


@dataclass
class StockData:
    identifier: str
    data: list[StockTableType]


def get_option_data(connection_string: str, database_name: str, table_name: str) -> OptionData:

    connection_string += database_name
    engine = create_engine(connection_string)
    #Base.metadata.create_all(engine)

    with sessionmaker(bind=engine)() as session:
        OptionQuery = build_OptionTable(table_name)
        data = session.query(OptionQuery).all()

    od: OptionData = OptionData(identifier=table_name, data=data)
    return od


def get_stock_data(connection_string: str, database_name: str, table_name: str) -> StockData:

    connection_string += database_name
    engine = create_engine(connection_string)
    #Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        StockQuery = build_StockTable(table_name)
        data = session.query(StockQuery).all()

    sd: StockData = StockData(identifier=table_name, data=data)
    return sd

    
def get_database_names(connection_string: str) -> dict[str, list[None]]:

    connection_string += 'master'
    engine = create_engine(connection_string)

    sql_server = {}
    with engine.connect() as conn:
        result = conn.execute(text('SELECT name FROM sys.databases'))
        for database in result:
            opt = '_OPT_' in database[0]
            stk = '_STK' in database[0]
            if opt or stk:
                sql_server[database[0]] = []
    
    return sql_server


def get_table_names(connection_string: str, sql_server: str) -> dict[str, list[str]]:
    
    for database in sql_server.keys():
        tmp_con_str = connection_string + database
        engine = create_engine(tmp_con_str)

        with engine.connect() as conn:
            result = conn.execute(text('SELECT name FROM sys.tables'))
            for table in result:
                sql_server[database].append(table[0])
    
    return sql_server


def filter_option_tables(sql_server: str) -> (dict[str, list[str]], dict[str, list[str]]):

    tmp_sql_server_stk, tmp_sql_server_opt = defaultdict(list), defaultdict(list)
    if DataCreationConfig.HISTORY_ONLY:
        cutoff_date_max = datetime.now() - timedelta(days=5)
        cutoff_date_max = cutoff_date_max.timestamp()
    else:
        cutoff_date_max = datetime(2099, 12, 31).timestamp()

    cutoff_date_min = datetime.strptime(DataCreationConfig.START_DATE, '%Y-%m-%d').timestamp()

    for database, tables in sql_server.items():
        for table in tables:
            if '_OPT_' in table:
                date = datetime.strptime(table.split('_')[2], '%d%b%y').timestamp()
                if cutoff_date_min <= date <= cutoff_date_max:
                    tmp_sql_server_opt[database].append(table)
            if '_STK' in database:
                tmp_sql_server_stk[database].append(table)
    
    return tmp_sql_server_stk, tmp_sql_server_opt


def load_sql_credentials() -> str:
    credentials_path = os.path.join(os.path.dirname(__file__), DataCreationConfig.SQL_CREDENTIALS_JSON_PATH)

    with open(credentials_path, 'r') as file:
        credentials = json.load(file)

    credentials['server'] = credentials['server'].replace('/', '\\')
    
    conn_str = f'mssql+pyodbc://{credentials['user']}:{credentials['password']}@{credentials['server']}?driver=ODBC+Driver+17+for+SQL+Server&database='

    return conn_str


def process_stock_data(conn_str: str, sql_server_stk: dict[str, list[str]]):
    stk_dfs = {}

    database = 'Data_STK'

    for i, table in enumerate(sorted(sql_server_stk[database]), start=1):
        if table == 'Query_Constituents':
            continue

        try:
            data = get_stock_data(conn_str, database, table)
        except InvalidRequestError as e:
            tprint(f'Invalid request error: {e}')
            continue
    
        if not data.data:
            continue

        stk_df = stk_data_to_polars(data)

        stk_dfs[table.split('_')[0]] = stk_df

    return stk_dfs


def safe_df_dumps(data_export_path: str, option_dfs: dict[str, pl.DataFrame], database: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        arrow_files = []

        for table, df in option_dfs.items():
            filename = f'{table}.arrow'
            filepath = os.path.join(tmpdir, filename)
            df.write_ipc(filepath)
            arrow_files.append(filename)
            
        if DataCreationConfig.SINGLE_SAVE:
            single_save_dir = os.path.join(data_export_path, database)
            if not os.path.exists(single_save_dir):
                os.mkdir(single_save_dir)

            for item in arrow_files:
                s = os.path.join(tmpdir, item)
                d = os.path.join(single_save_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
        
        if DataCreationConfig.MULTI_SAVE:
            output_path = os.path.join(data_export_path, f'{database}.tar.xz')

            try:
                command = [
                    'tar.exe',
                    '-cJf',
                    output_path,
                    *arrow_files
                ]
                subprocess.run(command, check=True, cwd=tmpdir)
                tprint(f'Successfully compressed {len(option_dfs)} DataFrames to {output_path}')

            except subprocess.CalledProcessError as e:
                tprint(f'Error during tar execution: {e}')
            except FileNotFoundError:
                tprint('tar.exe not found.')


def process_option_data(conn_str: str, sql_server_opt: dict[str, list[str]], stk_dfs: dict[str, pl.DataFrame]):
    option_dfs: dict[str, pl.DataFrame] = {}

    ordered_tables = sorted(sql_server_opt.keys(), key=lambda x: datetime.strptime(x.split('_')[2], '%b%y'))

    for database in ordered_tables:
        tables = sql_server_opt[database]
        t0 = perf_counter_ns()
        tprint(f'Double tables to queue: {set(x for x in tables if tables.count(x) > 1)}' )
        for i, table in enumerate(tables, start=1):
            t0 = perf_counter_ns()
            try:
                data = get_option_data(conn_str, database, table)
            except InvalidRequestError as e:
                tprint(f'Invalid request error: {e}')
                tprint(f'Requested opt data for {database} {table}')
                tprint(' - - - ')
                continue

            if not data.data:
                continue
            t1 = perf_counter_ns()

            tprint(f'Requesting opt data for {table} took {(t1-t0)/ 1e6} ms.')

            polars_df = option_data_to_polars(data, stk_dfs[table.split('_')[0]], table)

            if table in option_dfs.keys():
                tprint(f'{table} already exists as key.')

            option_dfs[table] = polars_df

            t2 = perf_counter_ns()
            tprint(f'Creating table for {table} took {(t2 - t1) / 1e6} ms.')
            tprint(' - - - ')

        if option_dfs.keys():
   
            t1 = perf_counter_ns()
            tprint(f'Received whole data for {database} in {((t1-t0)/1e9):.2f} seconds.')
            
            tprint(f'Exporting data for {database}...')
            data_export_path = os.path.join(os.path.dirname(__file__), DataCreationConfig.EXPORT_DIR)
            if not os.path.exists(data_export_path):
                os.makedirs(data_export_path)

            safe_df_dumps(data_export_path, option_dfs, database)

            t2 = perf_counter_ns()
            tprint(f'Created export files in {((t2-t1)/1e9):.2f} seconds.')
            
            option_dfs = {} 


def tprint(*args):

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : {' '.join(args)}")


def controller():
    conn_str: str = load_sql_credentials()

    sql_server: dict[str, list[None]] = get_database_names(conn_str)
    sql_server: dict[str, list[str]] = get_table_names(conn_str, sql_server)

    fot = filter_option_tables(sql_server)
    sql_server_stk: dict[str, list[str]] = fot[0]
    sql_server_opt: dict[str, list[str]] = fot[1]

    stk_dfs: dict[str, pl.DataFrame] = process_stock_data(conn_str, sql_server_stk)

    process_option_data(conn_str, sql_server_opt, stk_dfs)
    

    


    
