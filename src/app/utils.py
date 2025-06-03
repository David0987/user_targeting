import logging
from pyspark.sql.functions import col, lit, to_date
from pyspark.sql import DataFrame
from datetime import datetime, timedelta

# Assuming load_data and downsample_split will be imported correctly when used
# This requires main.py to pass SparkSession to these functions if they need it,
# or they need to get it from global scope (less ideal).
# load_data is in data_loader, downsample_split is in data_processing
# These utils will be called from main.py, which has access to spark.
# We might need to pass 'spark' to these utility functions.

# The original main.py calls:
# ratio_dict = calculate_positive_ratio(sc, Production_list, Starting_Day, Data_Path)
# train_df, validation_df, test_df = rolling_window(
#     Starting_Day, subcat, ratio_dict, Rolling_Days, Profile_Features,
#     Data_Path, NP_Ratio
# )
# And load_data was: load_data(spark, data_path, day, subcat, profile_features)
# So, calculate_positive_ratio needs spark, not just sc.
# And rolling_window also needs spark.

from .data_loader import load_data
from .data_processing import downsample_split

def calculate_positive_ratio(spark, production_list, starting_day, data_path_template, profile_features):
    """
    Calculates the ratio of positive to negative samples for each category.
    For simplicity, this version considers data only for the 'starting_day'.
    A more complex version might look at a window or average over time.
    The 'ratio' calculated here is P/N.
    """
    logging.info(f"Calculating positive ratio for production list: {production_list} on {starting_day}")
    ratio_dict = {}

    for subcat in production_list:
        logging.info(f"Calculating ratio for subcategory: {subcat}")
        try:
            # Load data for the specific day and subcategory
            # Note: load_data expects 'day' string, and profile_features.
            # The original load_data in data_processing.py selected 'label_' + subcat.
            # Our refactored load_data in data_loader.py takes subcat to handle this.
            data_df = load_data(spark, data_path_template, starting_day, subcat, profile_features)

            if data_df is None or data_df.rdd.isEmpty():
                logging.warning(f"No data found for {subcat} on {starting_day}. Setting ratio to 0.")
                ratio_dict[subcat] = 0.0
                continue

            positive_count = data_df.filter(col('label') == 1).count()
            negative_count = data_df.filter(col('label') == 0).count()

            if negative_count > 0:
                ratio = positive_count / negative_count
            elif positive_count > 0:
                # All positive, no negative, use a large ratio (or handle as per business logic)
                ratio = float('inf')
            else:
                ratio = 0.0 # No positive or negative samples

            ratio_dict[subcat] = ratio
            logging.info(f"Ratio for {subcat}: P={positive_count}, N={negative_count}, P/N Ratio={ratio}")

        except Exception as e:
            logging.error(f"Error calculating ratio for {subcat}: {e}", exc_info=True)
            ratio_dict[subcat] = 0.0 # Default ratio on error

    return ratio_dict


def daterange(start_date_str, num_days):
    """Generate a list of date strings in YYYY-MM-DD format."""
    base_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    date_list = []
    for i in range(num_days):
        day = base_date - timedelta(days=i)
        date_list.append(day.strftime('%Y-%m-%d'))
    return date_list

def rolling_window(spark, starting_day_str, subcat, ratio_dict_for_subcat_downsample,
                   rolling_days, profile_features, data_path_template, np_ratio_for_downsample):
    """
    Loads data for a rolling window of days, unions them, and then performs downsampling and splitting.
    'ratio_dict_for_subcat_downsample' is the P/N ratio from calculate_positive_ratio.
    'np_ratio_for_downsample' is the target N/P ratio for the training data.
    The 'fraction' in negative_df.sample in downsample_split is:
    target_neg_count / original_neg_count_in_unbalanced_train
    target_neg_count = positive_count_in_unbalanced_train * np_ratio_for_downsample
    So, fraction = (positive_count_in_unbalanced_train * np_ratio_for_downsample) / original_neg_count_in_unbalanced_train
                 = (P_u / N_u) * np_ratio_for_downsample
                 = ratio_dict_for_subcat_downsample * np_ratio_for_downsample
    This assumes ratio_dict_for_subcat_downsample is P/N of the unbalanced training set (or close to it).
    """
    logging.info(f"Processing rolling window for {subcat} ending on {starting_day_str} for {rolling_days} days.")

    all_days_df = None
    days_to_load = daterange(starting_day_str, rolling_days)
    logging.info(f"Loading data for days: {days_to_load}")

    for day_str in days_to_load:
        try:
            daily_df = load_data(spark, data_path_template, day_str, subcat, profile_features)
            if daily_df is not None and not daily_df.rdd.isEmpty():
                if all_days_df is None:
                    all_days_df = daily_df
                else:
                    # Ensure columns are exactly the same for union
                    # This should be handled by load_data returning consistent schema
                    all_days_df = all_days_df.unionByName(daily_df.select(all_days_df.columns))
            else:
                logging.warning(f"No data for {subcat} on {day_str}.")
        except Exception as e:
            logging.error(f"Error loading data for {subcat} on {day_str}: {e}", exc_info=True)

    if all_days_df is None or all_days_df.rdd.isEmpty():
        logging.error(f"No data loaded for any day in the window for {subcat}. Cannot proceed.")
        # Return empty DataFrames with expected schema (or handle as per pipeline requirements)
        # For now, let's signal error by returning Nones, main should handle this.
        # Or, create empty DFs with schema from a sample successful load_data call if possible.
        # This is complex, for now returning None. Main.py will need to check.
        # However, downsample_split expects a DataFrame.
        # Let's create empty DFs if all_days_df is None
        # Need schema. This is an issue.
        # For now, let's assume load_data if successful provides the schema.
        # If all_days_df is None, we can't call downsample_split.
        # The placeholder in main.py creates dummy DFs. We should do something similar if this fails.
        logging.error("Returning None for train/val/test due to no data in rolling window.")
        return None, None, None # Main needs to handle this robustly.

    all_days_df = all_days_df.distinct() # Remove exact duplicates across days if any
    all_days_df.persist()
    logging.info(f"Total rows loaded for {subcat} over {rolling_days} days (ending {starting_day_str}): {all_days_df.count()}")

    # The 'ratio' parameter for downsample_split is ratio_dict[subcat] from calculate_positive_ratio.
    # This 'ratio' is used as: negative_df.sample(fraction = ratio * np_ratio)
    # So, this 'ratio' should be P/N of the data being fed to downsample_split.
    # Let's recalculate P/N for all_days_df before splitting.
    # This makes calculate_positive_ratio's output less directly used by downsample_split's 'ratio' param,
    # which is confusing. The 'ratio_dict' passed to downsample_split should be the P/N of its input data.

    # Original plan: pass ratio_dict[subcat] (P/N from single day) to downsample_split.
    # The downsample_split function takes `ratio_dict` and `subcat`.
    # Let's stick to the original signature of downsample_split for now.
    # It expects `ratio_dict` (the full dict) and `subcat`.
    # And `np_ratio` (the target N/P for training).
    # So, downsample_split will use `ratio_dict[subcat]` as the 'ratio' factor.
    # This `ratio_dict[subcat]` is P/N of starting_day data.
    # And `np_ratio` is `NP_Ratio` from config.

    # If ratio_dict_for_subcat_downsample is P/N of the input data (all_days_df), then:
    # fraction = (P_all_days / N_all_days) * np_ratio_for_downsample
    # This seems more logical for the 'ratio' parameter in the sample call.
    # Let's assume downsample_split expects a dict and subcat to lookup the P/N of the input data.
    # So, we should compute P/N for all_days_df and pass that.

    current_positive_count = all_days_df.filter(col('label') == 1).count()
    current_negative_count = all_days_df.filter(col('label') == 0).count()
    current_pn_ratio = 0.0
    if current_negative_count > 0:
        current_pn_ratio = current_positive_count / current_negative_count
    elif current_positive_count > 0:
        current_pn_ratio = float('inf')

    # The downsample_split function expects a dictionary {subcat: ratio}
    # So we pass a new dict for the current subcat with its actual P/N ratio from the window.
    actual_pn_ratio_for_downsample_input = {subcat: current_pn_ratio}

    logging.info(f"Data for {subcat} (rolling window) has P/N ratio: {current_pn_ratio}")
    logging.info(f"Calling downsample_split for {subcat} with input P/N ratio {current_pn_ratio} and target N/P ratio {np_ratio_for_downsample}")

    train_df, validation_df, test_df = downsample_split(
        all_days_df,
        actual_pn_ratio_for_downsample_input, # This is the P/N ratio of `all_days_df`
        subcat,                            # Key to lookup in the dict
        np_ratio_for_downsample            # This is the target N/P for training output
    )

    all_days_df.unpersist()
    return train_df, validation_df, test_df
