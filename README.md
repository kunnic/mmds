> [!NOTE]
> **Acknowledgements**
>
> This project's implementation is inspired by the design of Apache Spark's `spark.ml`, including its OOP structures and design patterns.
>
> Apache Spark is licensed under the Apache License, Version 2.0 (January 2004).

# PCY Algorithm Implementation with PySpark

The Park–Chen–Yu (PCY) algorithm is implemented here, inspired by the structure of Apache Spark's `spark.ml.fpm`. The project is organized for readability and maintainability.

## Library version

- Apache Spark 3.5.4

```
pyspark-pcy-algorithm/
├── data/
│   └── baskets.csv
├── notebooks/
│   └── task02c.ipynb
├── .gitignore
├── LICENSE
├── pcy.py
└── README.md
```

## Installation

**IMPORTANT:** This project requires a local Spark installation to run. Please adapt the code to your environment as needed.

You can use this project by cloning the repository and installing the requirements, or by downloading the `pcy.py` file and using it directly.

## Usage

1. **Load the library as usual**
```python
from pcy import PCY, PCYModel
```

2. The `baskets.csv` file should be loaded into a Spark `DataFrame` with the following schema:
```
["id", "date", "item_list"]
```

3. Create a PCY model with `min_support`, `min_confidence`, and `num_buckets` parameters:
```python
model = PCY(min_support=<your_support>, min_confidence=<your_confidence>, num_buckets=<your_bucket_number>)
# model.fit(baskets_df)
```