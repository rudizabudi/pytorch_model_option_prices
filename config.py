from dataclasses import dataclass
from enum import StrEnum


@dataclass
class DataCreationConfig:
    class OutputType(StrEnum):
        BACKUP = 'BACKUP'
        TRAINING = 'TRAINING'

    OUTPUT_TYPE: OutputType = OutputType.TRAINING  # BACKUP | TRAINING
    TRAINING_EXPORT_DIR: str = 'training_data'  # directory to save training data
    BACKUP_EXPORT_DIR: str = 'backup_data'  # directory to save backup data

    HISTORY_ONLY: bool = True  # only use historical data
    START_DATE: str = '2024-08-01'  # YYYY-MM-DD start date for historical data to process
    REFRESH_DATA: bool = True  # only process data for tables that dont already exist

    SQL_CREDENTIALS_JSON_PATH: str = 'sql_credentials.json'

    XLS_RISK_FREE_RATE_URL: str = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xmlview?data=daily_treasury_yield_curve&field_tdr_date_value='

    TRAIN_IGNORE_FOLDERS: str = 'Vergleich'  # comma separated

    FMP_API_KEY: str = 'rbfZHTUCSk52aCXELDmES40UraSXKgJL'  # financialmodelingprep API Key


@dataclass
class TrainingConfig:
    class TrainingHardware(StrEnum):
        CPU = 'CPU'
        CUDA = 'CUDA'

    TRAINING_HARDWARE: TrainingHardware = TrainingHardware.CPU
    TRAINING_DATA_PATH: str = 'training_data'

    FEATURES: tuple[str] = ('time_to_expiry_days',  'callput', 'strike', 'h', 'l', 'o', 'c', 'risk_free_rate', 'ul_h', 'ul_l', 'ul_o', 'ul_c')
    TARGET: tuple[str] = ('h', 'l', 'o', 'c')

    VAL_SIZE: float = 0.2  # relative size of val data
    TRAIN_BATCH_SIZE: int = 32
    VAL_BATCH_SIZE: int = 64

    TRAINING_EPOCHS: int = 20

    MODEL_SAVE_NAME: str = 'full_model.pth'


