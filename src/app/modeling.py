from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier  
from pyspark.ml import Pipeline

def assemble_model(train_df, model_params):
    """Assemble GBT classification model pipeline."""
    feature_cols = [c for c in train_df.columns if c != 'label']

    # Assemble feature columns into a vector
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Initialize GBT classifier
    gbt = GBTClassifier(
        labelCol="label",
        featuresCol=assembler.getOutputCol(),
        maxMemoryInMB=2048,
        cacheNodeIds=True,
        **model_params
    )

    # Assemble pipeline  
    pipeline = Pipeline(stages=[assembler, gbt])

    # Train model
    model = pipeline.fit(train_df)

    return model

def save_model(model, model_path):
    """Save trained model."""  
    model.write().overwrite().save(model_path) 