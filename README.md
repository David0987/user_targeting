# User Targeting ML Pipeline

This project implements an MLOps pipeline for user targeting based on Spark ML.
It processes user data, trains models for different categories, evaluates them, and saves the models.

## Project Structure

```
.
├── LICENSE
├── README.md
├── requirements.txt
├── scripts/               # For standalone or utility scripts (currently empty)
├── src/
│   └── app/
│       ├── __init__.py    # (Should be added to make 'app' a package)
│       ├── config.py      # Configuration variables (paths, model params)
│       ├── data_loader.py # Data loading functions
│       ├── data_processing.py # Data transformation and splitting functions
│       ├── main.py        # Main executable script for the pipeline
│       ├── model_evaluator.py # Model evaluation functions
│       ├── modeling.py    # Model training and saving functions
│       └── utils.py       # Utility functions (ratio calculation, rolling window)
└── tests/
    ├── __init__.py      # (Should be added to make 'tests' discoverable)
    └── test_data_processing.py # Unit tests for data_processing module
```

## Setup

1.  **Clone the repository:**
    `git clone <repository_url>`
    `cd user_targeting`

2.  **Install dependencies:**
    Ensure you have Python and Spark installed.
    Install PySpark and other Python dependencies:
    `pip install -r requirements.txt`
    (Note: `pytest` is needed to run tests: `pip install pytest`)


3.  **Configure paths:**
    Update the path configurations in `src/app/config.py` (e.g., `Data_Path`, `Model_Save_Path`) to point to your data and model storage locations. Consider using environment variables for these as commented in the file.

## Running the Pipeline

The main pipeline can be executed using `main.py`. You can specify the starting day and production categories via command-line arguments.

```bash
# Example:
python src/app/main.py --starting_day YYYY-MM-DD --production_list category_A category_B
```

Default values (if not provided):
-   `--starting_day`: `2020-10-31`
-   `--production_list`: `['category1', 'category2']`

The script will output logs to the console. Models and summaries will be saved to the paths specified in `config.py`.

## Running Tests

To run the unit tests:
```bash
# Ensure PYTHONPATH includes the src directory if running from the project root
export PYTHONPATH=$(pwd)/src:$PYTHONPATH
pytest tests/
```

## MLOps Considerations

-   **Configuration:** Managed via `config.py`. Key paths can be overridden by environment variables (requires code modification to implement `os.environ.get`).
-   **Logging:** Implemented throughout the pipeline using Python's `logging` module.
-   **Modularity:** Code is broken down into modules by functionality (data loading, processing, modeling, evaluation, utils).
-   **Testing:** Basic unit tests are in the `tests/` directory (e.g., for `downsample_split`).
-   **Dependencies:** Listed in `requirements.txt`.
-   **Entry Point:** `main.py` is configurable via command-line arguments (`argparse`).
