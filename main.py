from sql_request_data import controller
from train_model import startup
from config import DataCreationConfig


def create_training_data():
    controller()


def train_model():
    startup()


if __name__ == "__main__":
    create_training_data()
    #train_model()

#Todo: Normalize data for stock splits