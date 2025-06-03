import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from src.app.data_processing import downsample_split # Assuming src.app is in PYTHONPATH or adjusted

@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-spark-session")
        .getOrCreate()
    )
    yield spark
    spark.stop()

def test_downsample_split(spark):
    # Create dummy data
    schema = "user_id STRING, label INT, feature1 INT"
    data = []
    for i in range(100): # 100 negative samples
        data.append((f"user_neg_{i}", 0, i))
    for i in range(20): # 20 positive samples
        data.append((f"user_pos_{i}", 1, i))

    source_df = spark.createDataFrame(data, schema=schema)
    source_df.persist() # Persist for count

    positive_count = source_df.filter(col('label') == 1).count() # Expected 20
    negative_count = source_df.filter(col('label') == 0).count() # Expected 100

    # Dummy ratio_dict and np_ratio
    # ratio_dict usually comes from calculate_positive_ratio
    # If positive is 20 and negative is 100, original ratio P/N = 0.2.
    # If we want NP_Ratio (Negative to Positive in training) = 1.0,
    # then the fraction for negative_df.sample should be:
    # (target_neg_count / original_neg_count)
    # target_neg_count = positive_count_in_train * np_ratio
    # This is tricky because downsample_split splits train/test first.
    # Let's simplify for the test: assume ratio_dict provides a direct sampling fraction
    # for the negative samples *after* the initial 95/5 split.

    # After 95/5 split (approx): 19 positive, 95 negative in initial train.
    # If np_ratio = 1.0, we want ~19 negative samples in the final train_balanced_df.
    # So, fraction for negative_df.sample would be roughly 19/95 = 0.2.
    # Let ratio_dict provide this 0.2.

    # Let's make it more direct for the test:
    # Positive examples in train_df (before balancing): approx 0.95 * 20 = 19
    # Negative examples in train_df (before balancing): approx 0.95 * 100 = 95
    # If np_ratio = 1.0, we want approx 19 negative samples in the final train_df.
    # The fraction to sample from negative_df is (target_negative_samples / total_negative_samples_in_train_unbalanced)
    # So, fraction = (positive_samples_in_train_unbalanced * np_ratio) / total_negative_samples_in_train_unbalanced
    # fraction = (19 * 1.0) / 95 = 0.2

    # The 'ratio' in downsample_split is ratio_dict[subcat]. This is confusingly named.
    # It seems 'ratio' in downsample_split is meant to be the *target* P/N ratio for *positive* sampling
    # and np_ratio is for *negative* sampling. The current implementation uses:
    # negative_df.sample(withReplacement=False, fraction=ratio*np_ratio, seed=42)
    # This means ratio_dict[subcat] * np_ratio is the final fraction for negative sampling.
    # Let's assume ratio_dict[subcat] = 0.1 and np_ratio = 2.0, so fraction = 0.2.
    # This would mean we take 20% of the negative samples.

    ratio_dict_test = {'category_test': 0.1} # This 'ratio' seems to be a factor
    np_test_ratio = 2.0 # Target N/P ratio in training set
                        # The sampling fraction becomes ratio_dict_test['category_test'] * np_test_ratio = 0.2

    train_df, validation_df, test_df = downsample_split(source_df, ratio_dict_test, 'category_test', np_test_ratio)

    # Assertions
    # 1. Columns
    expected_cols = ["user_id", "label", "feature1"]
    assert all(c in train_df.columns for c in expected_cols)
    assert all(c in validation_df.columns for c in expected_cols)
    assert all(c in test_df.columns for c in expected_cols)

    # 2. Test set size (approx 5% of original)
    assert abs(test_df.count() - (source_df.count() * 0.05)) < 5 # Allow some variance

    # 3. Train + Validation should be the remaining 95%
    # And after downsampling, it will be smaller.
    # Original train_full_df count = approx 120 * 0.95 = 114
    # Positive in train_full_df = approx 20 * 0.95 = 19
    # Negative in train_full_df = approx 100 * 0.95 = 95
    # Negative samples after downsampling (fraction = 0.1 * 2.0 = 0.2)
    # Expected negative in train_balanced_df = approx 95 * 0.2 = 19
    # Expected total in train_balanced_df = 19 (positive) + 19 (negative) = 38

    train_val_balanced_count = train_df.count() + validation_df.count()
    # Allow some variance due to randomSplit and sample
    assert abs(train_val_balanced_count - 38) < 10

    # 4. In train_df (after 80/20 split of balanced):
    #    - N/P ratio should be close to np_test_ratio (2.0)
    #    - Positive count should be approx 38 * 0.8 * (P / (P+N))
    #      P/(P+N) ratio in balanced set is P / (P + P*np_test_ratio) = 1 / (1 + np_test_ratio)
    #      So, P/(P+N) = 1 / (1+2) = 1/3.
    #      Expected positive in train_df = (approx 38 * 0.8) * (1/3) = 30.4 * 1/3 = 10.13
    #      Expected negative in train_df = (approx 38 * 0.8) * (2/3) = 30.4 * 2/3 = 20.26

    train_pos_count = train_df.filter(col('label') == 1).count()
    train_neg_count = train_df.filter(col('label') == 0).count()

    if train_pos_count > 0: # Avoid division by zero
        assert abs((train_neg_count / train_pos_count) - np_test_ratio) < 0.5 # Allow some variance

    source_df.unpersist()
    train_df.unpersist()
    validation_df.unpersist()
    test_df.unpersist()
