from pyspark.ml.evaluation import BinaryClassificationEvaluator

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
