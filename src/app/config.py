# Global configuration variables

DATABASE = 'ecommerce'

Starting_Day = '2020-10-31'
Rolling_Days = 10 
Production_list = ['category1', 'category2']  # Replace with actual categories

Data_Path = "/path/to/feature/data/grass_region=TW/grass_date={Day}"  # Consider using an environment variable: os.environ.get("DATA_PATH_TEMPLATE")
Model_Save_Path = "/path/to/save/model/{Subcat}/{Day}/model"  # Consider using an environment variable: os.environ.get("MODEL_SAVE_PATH_TEMPLATE")
Model_Summary_Path = "/path/to/save/model_summary/"  # Consider using an environment variable: os.environ.get("MODEL_SUMMARY_PATH")

Model_Complexity = {
    'maxDepth': 12, 
    'maxIter': 210,
    'maxBins': 32,
    'minInstancesPerNode': 512
}

NP_Ratio = 2.0  # Negative to Positive Ratio for downsampling

Profile_Features = [
    'user_id', 'age_range', 'gender'  # Replace with actual profile features
] 