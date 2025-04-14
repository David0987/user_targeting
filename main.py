import time
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from config import *
from data_processing import load_data, downsample_split, union_df  
from modeling import assemble_model, evaluate_model, save_model
from utils import calculate_positive_ratio, rolling_window

def main():
    # Create Spark context and session
    SPARK_CONF = SparkConf()
    sc = SparkContext(appName='ml_pipeline', conf=SPARK_CONF)
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    # Use the specified database
    spark.sql(f'use {DATABASE}')

    # Calculate positive ratio for each category
    tic = time.time()
    ratio_dict = calculate_positive_ratio(sc, Production_list, Starting_Day, Data_Path) 
    toc = time.time()
    print(f'Time to calculate ratio (hours): {(toc - tic)/3600}')

    # Train and evaluate models for each category
    for subcat in Production_list:
        train_df, validation_df, test_df = rolling_window(
            Starting_Day, subcat, ratio_dict, Rolling_Days, Profile_Features, 
            Data_Path, NP_Ratio
        )

        model = assemble_model(train_df, Model_Complexity)

        model_path = Model_Save_Path.format(Subcat=subcat, Day=Starting_Day)  
        save_model(model, model_path)

        train_metrics = evaluate_model(model, train_df)
        validation_metrics = evaluate_model(model, validation_df)
        test_metrics = evaluate_model(model, test_df)

        # TODO: Save model summary

        # Clear cache to free up memory  
        spark.catalog.clearCache()

if __name__ == "__main__":
    main() 