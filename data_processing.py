from pyspark.sql.functions import col
from pyspark.storagelevel import StorageLevel

def load_data(spark, data_path, day, subcat, profile_features):
    """Load and preprocess data for a given day and category."""
    # Read data from parquet files
    data_df = spark.read.parquet(data_path.format(Day=day))

    # Select relevant columns
    columns_to_keep = ['user_id', 'label_' + subcat] + profile_features
    data_df = data_df.select(*columns_to_keep)

    # Rename label column
    data_df = data_df.withColumnRenamed('label_' + subcat, 'label')

    # Cast label to integer
    data_df = data_df.withColumn('label', col('label').cast('integer'))  

    # Fill missing values
    data_df = data_df.fillna({'age_range': 'unknown', 'gender': 'unknown'})

    # Repartition by user_id for better performance
    data_df = data_df.repartition('user_id')

    return data_df

def downsample_split(data_df, ratio_dict, subcat, np_ratio):
    """Downsample majority class and split into train, validation and test sets."""
    # Split into train (95%) and test (5%)
    train_df, test_df = data_df.randomSplit([0.95, 0.05], seed=42)

    # Separate positive and negative examples in train set  
    positive_df = train_df.filter(col('label') == 1)
    negative_df = train_df.filter(col('label') == 0)

    # Downsample negative examples
    ratio = ratio_dict[subcat]
    negative_sample_df = negative_df.sample(withReplacement=False, fraction=ratio*np_ratio, seed=42)

    # Combine positive and downsampled negative examples
    train_balanced_df = positive_df.union(negative_sample_df)

    # Split balanced train set into train (80%) and validation (20%)  
    train_df, validation_df = train_balanced_df.randomSplit([0.8, 0.2], seed=42)

    # Persist dataframes in memory
    train_df.persist(StorageLevel.MEMORY_AND_DISK)  
    validation_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df.persist(StorageLevel.MEMORY_AND_DISK)

    return train_df, validation_df, test_df

def union_df(df1, df2):
    """Union two dataframes with common columns."""
    common_columns = df1.columns
    unioned_df = df1.select(common_columns).union(df2.select(common_columns))  
    return unioned_df.repartition('user_id') 