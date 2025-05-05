from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import os
import polars as pl
from pyarrow import ipc, OSFile
from sqlalchemy import create_engine, String, Float, text
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker
from sqlalchemy.exc import InvalidRequestError, SAWarning
from sqlalchemy.ext.declarative import declarative_base
from time import perf_counter_ns
import warnings

from convert_to_df import option_data_to_polars, stk_data_to_polars
from config import DataCreationConfig

if os.getenv('DEV_VAR') == 'rudizabudi':
    DataCreationConfig.SQL_CREDENTIALS_JSON_PATH = 'sql_credentials_dev.json'

    LOGGING = False
    if LOGGING:
        import logging

        logging.basicConfig()
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

warnings.filterwarnings('ignore', category=SAWarning)

type OptionTableType = 'OptionTableType'
type StockTableType = 'StockTableType'

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
            return (f'<StockTable(date={self.date}, h={self.h},'
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
    # Base.metadata.create_all(engine)

    with sessionmaker(bind=engine)() as session:
        option_query = build_OptionTable(table_name)
        data = session.query(option_query).all()

    od: OptionData = OptionData(identifier=table_name, data=data)

    return od


def get_stock_data(connection_string: str, database_name: str, table_name: str) -> StockData:

    connection_string += database_name
    engine = create_engine(connection_string)
    # Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        stock_query = build_StockTable(table_name)
        data = session.query(stock_query).all()

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


def get_table_names(connection_string: str, sql_server: dict[str, list[None]]) -> dict[str, list[None | str]]:
    
    for database in sql_server.keys():
        tmp_con_str = connection_string + database
        engine = create_engine(tmp_con_str)

        with engine.connect() as conn:
            result = conn.execute(text('SELECT name FROM sys.tables'))
            for table in result:
                sql_server[database].append(table[0])
    
    return sql_server


def filter_option_tables(sql_server: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:

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

    existing_tables = []
    if DataCreationConfig.REFRESH_DATA:
        subfolder = DataCreationConfig.TRAINING_EXPORT_DIR if DataCreationConfig.OUTPUT_TYPE == 'TRAINING' else DataCreationConfig.BACKUP_EXPORT_DIR

        for root, _, files in os.walk(subfolder):
            for file in files:
                if file.endswith('.arrow'):
                    if not any(map(lambda x: x in root, DataCreationConfig.TRAIN_IGNORE_FOLDERS.split(','))):
                        existing_tables.append(file.rstrip('.arrow'))

    for database in tmp_sql_server_opt.keys():
        tmp_list = []
        for table in tmp_sql_server_opt[database]:
            if table not in existing_tables:
                tmp_list.append({'query_db': database, 'table': table})
        tmp_sql_server_opt[database] = tmp_list

    # Handles split months eg Jan25 and Jan25_1
    opt_data_keys = list(tmp_sql_server_opt.keys())
    for i, lk in enumerate(opt_data_keys):
        to_add = [k for k in opt_data_keys[i + 1:] if lk.split('_')[2] == k.split('_')[2]]
        for add_key in to_add:
            tmp_sql_server_opt[lk].extend(tmp_sql_server_opt[add_key])
            tmp_sql_server_opt.pop(add_key)

    return tmp_sql_server_stk, tmp_sql_server_opt


def load_sql_credentials() -> str:
    credentials_path = os.path.join(os.path.dirname(__file__), DataCreationConfig.SQL_CREDENTIALS_JSON_PATH)

    with open(credentials_path, 'r') as file:
        credentials = json.load(file)

    credentials['server'] = credentials['server'].replace('/', '\\')
    
    conn_str = f'mssql+pyodbc://{credentials['user']}:{credentials['password']}@{credentials['server']}?driver=ODBC+Driver+17+for+SQL+Server&database='

    return conn_str


def process_stock_data(conn_str: str, sql_server_stk: dict[str, list[str]]) -> dict[str, pl.DataFrame]:
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


def write_df_to_file(polars_df: pl.DataFrame, database: str, table: str) -> None:

    subfolder = DataCreationConfig.TRAINING_EXPORT_DIR if DataCreationConfig.OUTPUT_TYPE == 'TRAINING' else DataCreationConfig.BACKUP_EXPORT_DIR

    output_path = os.path.join(os.path.dirname(__file__), subfolder, database)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_path = os.path.join(output_path, f'{table}.arrow')
    with OSFile(file_path, mode='wb') as file:
        with ipc.new_file(sink=file, schema=polars_df.to_arrow().schema, options=ipc.IpcWriteOptions(compression="zstd")) as writer:
            writer.write_table(polars_df.to_arrow())

    # To read the file back
    # with memory_map(file_path, 'r') as file:
    #     df = ipc.RecordBatchFileReader(file).read_all()
    #     df = pl.from_arrow(df)


def process_option_data(conn_str: str, sql_server_opt: dict[str, list[dict[str, str]]], stk_dfs: dict[str, pl.DataFrame]):

    ordered_tables = sorted(sql_server_opt.keys(), key=lambda x: datetime.strptime(x.split('_')[2], '%b%y'))

    for database in ordered_tables:
        tables = sql_server_opt[database]

        t0 = perf_counter_ns()
        for i, query_data in enumerate(tables, start=1):
            query_table = query_data['table']
            query_db = query_data['query_db']

            t1 = perf_counter_ns()
            try:
                data = get_option_data(conn_str, query_db, query_table)
            except InvalidRequestError as e:
                tprint(f'Invalid request error: {e}')
                tprint(f'Requested opt data for {query_db} {query_table}')
                tprint(' - - - ')
                continue

            if not data.data:
                continue

            t2 = perf_counter_ns()

            tprint(f' {i:^5}/{len(tables):^8} | {(i / len(tables)) * 100:.2f}% for {database}.')
            tprint(f'Requesting opt data for {query_table} took {(t2-t1) / 1e6:.0f} ms.')

            polars_df = option_data_to_polars(data, stk_dfs[query_table.split('_')[0]], query_table)

            t3 = perf_counter_ns()
            tprint(f'Creating dataframe for {query_table} took {(t3 - t2) / 1e6:.0f} ms.')

            write_df_to_file(polars_df, database, query_table)
            t4 = perf_counter_ns()
            tprint(f'Writing dataframe to file took {(t4 - t3) / 1e6:.0f} ms.')
            tprint(' - - - ')

        if 't4' in locals():
            tprint(f'Received whole data for {database} in {((t4 - t0) / 1e9):.2f} seconds.')


def tprint(*args):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} : {' '.join(args)}")


def controller():
    tprint(f'Started in mode {DataCreationConfig.OUTPUT_TYPE}.')
    conn_str: str = load_sql_credentials()

    sql_server: dict[str, list[None]] = get_database_names(conn_str)
    sql_server: dict[str, list[str]] = get_table_names(conn_str, sql_server)

    fot = filter_option_tables(sql_server)
    sql_server_stk: dict[str, list[str]] = fot[0]
    sql_server_opt: dict[str, list[str]] = fot[1]

    stk_dfs: dict[str, pl.DataFrame] = process_stock_data(conn_str, sql_server_stk)

    process_option_data(conn_str, sql_server_opt, stk_dfs)
    

    


    
