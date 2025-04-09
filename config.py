from dataclasses import dataclass

@dataclass
class DataCreationConfig:

    HISTORY_ONLY: bool = True  # only use historical data

    EXPORT_DIR: str = 'data'  # directory to save the data
    SINGLE_SAVE: bool = True  # save data in a single file
    MULTI_SAVE: bool = True  # save data in multiple files
    DELETE_PICKLE: bool = False  # only if compress_data is True

    SQL_CREDENTIALS_JSON_PATH: str = 'sql_credentials.json'
