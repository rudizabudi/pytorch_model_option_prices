from dataclasses import dataclass

@dataclass
class DataCreationConfig:

    HISTORY_ONLY: bool = True  # only use historical data
    START_DATE: str = '2025-02-01'  # YYYY-MM-DD start date for historical data to process

    EXPORT_DIR: str = 'data'  # directory to save the data
    SINGLE_SAVE: bool = True  # save data in a single file
    MULTI_SAVE: bool = True  # save data in multiple files
    DELETE_PICKLE: bool = False  # only if compress_data is True

    SQL_CREDENTIALS_JSON_PATH: str = 'sql_credentials.json'

    XLS_RISK_FREE_RATE_URL: str = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xmlview?data=daily_treasury_yield_curve&field_tdr_date_value='
