# ILT 2: Interacting with Data

How to interacting with data on Python. The notebook is available. Just walk through.

## Overview

This material covers the basics of data interaction using Python. It includes practical examples and exercises to help you understand how to manipulate and analyze data effectively.

## Contents

- **Introduction to Pandas**: Learn the basics of the Pandas library, including data structures like Series and DataFrame.
- **Data Cleaning**: Techniques for handling missing data, duplicates, and data transformation.
- **Data Visualization**: Using libraries like Matplotlib and Seaborn to create informative visualizations.
- **Statistical Analysis**: Performing basic statistical operations and hypothesis testing.

## Examples

### Loading Data

```python
import pandas as pd

# Load a CSV file into a DataFrame
df = pd.read_csv('data.csv')
print(df.head())
```

### Data Cleaning

```python
# Drop missing values
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)
```

### Data Visualization

```python
import matplotlib.pyplot as plt

# Create a simple line plot
plt.plot(df['column_name'])
plt.show()
```

### Statistical Analysis

```python
# Calculate mean and standard deviation
mean = df['column_name'].mean()
std_dev = df['column_name'].std()
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
```

Feel free to explore the notebook and try out the examples provided. Happy learning!
