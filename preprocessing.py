from helper.data_read_preparation import read_and_check_data
from helper.feature_engineering import feature_engineering
from helper.constant import PATH

def main():
    df = read_and_check_data(PATH)

    df_transformed = feature_engineering(df)
    print("Start Saving Result Feature Engineering!")
    df_transformed.to_csv("artifacts/df_transformed.csv")


if __name__ == "__main__":
    print("START RUNNING PIPELINE!")
    main()
    print("FINISH RUNNING PIPELINE!")