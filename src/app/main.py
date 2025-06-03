import time
import logging
import argparse
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from .config import *
from .data_loader import load_data
from .data_processing import downsample_split, union_df
from .modeling import assemble_model, save_model
from .model_evaluator import evaluate_model
from .utils import calculate_positive_ratio, rolling_window

# Placeholder function for model summary
def save_model_summary(model, metrics, path):
    """Placeholder function to save model summary and metrics."""
    print(f"TODO: Save model summary and metrics to {path}")
    print(f"Metrics: {metrics}")

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline for User Targeting")
    parser.add_argument("--starting_day", type=str, default='2020-10-31',
                        help="The starting day for data processing, format YYYY-MM-DD")
    parser.add_argument("--production_list", nargs='+', default=['category1', 'category2'],
                        help="List of production categories to process")
    args = parser.parse_args()
    """Main function to run the ML pipeline: loads data, trains models, evaluates, and saves them."""

    effective_starting_day = args.starting_day
    effective_production_list = args.production_list

    # Create Spark context and session
    SPARK_CONF = SparkConf()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Starting ML Pipeline with Starting Day: {effective_starting_day} and Production List: {effective_production_list}")
    sc = SparkContext(appName='ml_pipeline', conf=SPARK_CONF)
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    # Use the specified database
    spark.sql(f'use {DATABASE}')

    # Calculate positive ratio for each category
    tic = time.time()
    ratio_dict = calculate_positive_ratio(spark, effective_production_list, effective_starting_day, Data_Path, Profile_Features)
    toc = time.time()
    logging.info(f'Time to calculate ratio (hours): {(toc - tic)/3600}')

    # Train and evaluate models for each category
    for subcat in effective_production_list:
        try:
            logging.info(f"Starting processing for subcategory: {subcat}")
            train_df, validation_df, test_df = rolling_window(
                spark, effective_starting_day, subcat, ratio_dict, Rolling_Days, Profile_Features,
                Data_Path, NP_Ratio
            )
            if train_df is None: # Or check all three if necessary
                logging.warning(f"Skipping subcategory {subcat} due to issues in rolling_window data loading.")
                continue

            model = assemble_model(train_df, Model_Complexity)

            model_path = Model_Save_Path.format(Subcat=subcat, Day=effective_starting_day)
            save_model(model, model_path)

            train_metrics = evaluate_model(model, train_df)
            validation_metrics = evaluate_model(model, validation_df)
            test_metrics = evaluate_model(model, test_df)

            summary_path = Model_Summary_Path + f"{subcat}_{effective_starting_day}_summary.txt" # Example path
            all_metrics = {"train": train_metrics, "validation": validation_metrics, "test": test_metrics}
            save_model_summary(model, all_metrics, summary_path)
            logging.info(f"Successfully processed subcategory: {subcat}")
        except Exception as e:
            logging.error(f"Error processing subcategory {subcat}: {e}", exc_info=True)
        finally:
            logging.info(f"Finished processing for subcategory: {subcat}. Clearing cache.")
            # Clear cache to free up memory
            spark.catalog.clearCache()
    logging.info("ML Pipeline finished.")

if __name__ == "__main__":
    main()