from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier  
from pyspark.ml.evaluation import BinaryClassificationEvaluator
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

def evaluate_model(model, data_df):  
    """Evaluate model performance on a dataset."""
    predictions = model.transform(data_df)

    # Evaluate area under ROC
    roc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
    roc_auc = roc_evaluator.evaluate(predictions)  

    # Evaluate area under precision-recall curve  
    pr_evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
    pr_auc = pr_evaluator.evaluate(predictions)

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}

def save_model(model, model_path):
    """Save trained model."""  
    model.write().overwrite().save(model_path) 