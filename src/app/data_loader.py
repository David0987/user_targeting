from pyspark.sql.functions import col

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
