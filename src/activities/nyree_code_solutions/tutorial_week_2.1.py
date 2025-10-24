#2.1 - python structure
# using pathlib for determining path

import sys
print(sys.executable)

from pathlib import Path

# this file is src/activities/nyree_code_solutions/tutorial_activities.py
# the other file is src/activities/data/paralymics_raw.csv

# project_root = Path(__file__).parent.parent.parent
# paralympics_csv_path = project_root / 'activities' / 'data' / 'paralympics_raw.csv'
# paralympics_xlsx_path = project_root / 'activities' / 'data' / 'paralympics_all_raw.xlsx'

# print(paralympics_csv_path.exists())  # this should print True if the path is correct
# print(paralympics_xlsx_path.exists())  # this should print True if the path is correct



#2.2 - pandas-df

import pandas as pd

# paralympics_csv_file = pd.read_csv(paralympics_csv_path)
# print(paralympics_csv_df.head())

# paralympics_xlsx_file = pd.read_excel(paralympics_xlsx_path, sheet_name=0)
# print(paralympics_xlsx_df_1.head())



#2.3 - pandas-describe

def describe_dataframe(paralympics_csv_file: pd.DataFrame):
    """Prints descriptive information about the Paralympics DataFrame.
    
    Parameters:
        paralympics_csv_file (pd.DataFrame): The DataFrame containing Paralympics data.
        
        Returns:
            None
    """
    pd.set_option("display.max_columns", None)

    print("DataFrame shape (rows, columns):", paralympics_csv_file.shape)
    print("First 5 rows:", paralympics_csv_file.head())
    print("Last 5 rows:", paralympics_csv_file.tail())
    print("Column labels:", paralympics_csv_file.columns)
    print("Column data types:", paralympics_csv_file.dtypes)
    print("Info:", paralympics_csv_file.info())
    print("Descriptive statistics:", paralympics_csv_file.describe())

#if __name__ == "__main__":
    #project_root = Path(__file__).parent.parent.parent

    #paralympics_csv_path = project_root / 'activities' / 'data' / 'paralympics_raw.csv'
    #paralympics_xlsx_path = project_root / 'activities' / 'data' / 'paralympics_all_raw.xlsx'

    #paralympics_csv_file = pd.read_csv(paralympics_csv_path)
    #paralympics_xlsx_file = pd.read_excel(paralympics_xlsx_path, sheet_name=0)

    #print("CSV data description:")
    #describe_dataframe(paralympics_csv_file)
    #print("Excel data description:")
    #describe_dataframe(paralympics_xlsx_file)



#2.4 - identify missing values
# missing values are represented as NaN/None in pandas, empty strings are not considered missing values by default
    
def check_dataquality(df: pd.DataFrame):
    """Checks data quality and prints missing value information.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to check for missing values.
        
    Returns:
        None
    """
    missing_values = df.isnull().sum() # count missing values in each column
    print(f"Missing values in each column: {missing_values}")

    total_missing = missing_values.sum() # total missing values in the DataFrame
    print(f"Total missing values in DataFrame: {total_missing}")

    missing_rows = df[df.isnull().any(axis=1)] # rows with any missing values
    print(f"Rows with missing values:\n{missing_rows}")

    missing_columns = df.loc[:, df.isnull().any(axis=0)] # columns with any missing values
    print(f"Columns with missing values:\n{missing_columns}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent

    paralympics_csv_path = project_root / 'activities' / 'data' / 'paralympics_raw.csv'
    paralympics_xlsx_path = project_root / 'activities' / 'data' / 'paralympics_all_raw.xlsx'

    paralympics_csv_file = pd.read_csv(paralympics_csv_path)
    paralympics_xlsx_file = pd.read_excel(paralympics_xlsx_path, sheet_name=0)

    print("CSV data quality check:")
    check_dataquality(paralympics_csv_file)
    print("Excel data quality check:")
    check_dataquality(paralympics_xlsx_file)



#2.5 - introduction to charts with pandas.Dataframe.plot
    

#2.6 - plotting histograms and boxplots

def plot_histogram(df: pd.DataFrame, column: str):
    """Plots a histogram for a specified column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to plot the histogram for.
    
    Returns:
        None
    """
    import matplotlib.pyplot as plt

    #Check if the column exists in the DataFrame
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame.")
        return

    # Drop NaN values to avoid errors
    data = df[column].dropna()

    # Plot the histogram
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=10, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Show and/or save the plot
    plt.show()
    # plt.savefig(f"histogram_{column}.png")

#if __name__ == "__main__":
    #project_root = Path(__file__).parent.parent.parent
    #paralympics_csv_path = project_root / 'activities' / 'data' / 'paralympics_raw.csv'

    #df = pd.read_csv(paralympics_csv_path)
    #plot_histogram(df, "participants_f")


def plot_boxplot(df: pd.DataFrame, columns: list[str] = None):
    """Plots box plots for specified columns (or all numeric columns) in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the Paralympics data.
        columns (list[str], optional): A list of column names to plot box plots for.
                                       If None, all numeric columns will be plotted.
    
    Returns:
        None
    """

    import matplotlib.pyplot as plt

     # If no columns provided, select all numeric ones
    if columns is None:
        numeric_df = df.select_dtypes(include=['number'])
        print(f"Creating box plots for all numeric columns: {numeric_df.columns.tolist()}")
        numeric_df.plot(kind='box', figsize=(10, 6), grid=True)
    else:
        # Validate provided columns
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            print(f"Warning: These columns were not found in the DataFrame: {missing_cols}")

        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            print("No valid columns found to plot.")
            return

        print(f"Creating box plots for: {valid_cols}")
        df[valid_cols].plot(kind='box', subplots=True, layout=(1, len(valid_cols)), figsize=(6 * len(valid_cols), 5), grid=True)

    plt.suptitle("Box Plots of Paralympics Data", fontsize=14)
    plt.tight_layout()
    plt.show()




#2.7 - time series data
def plot_timeseries(df: pd.DataFrame, x_col: str = "year", y_cols: list[str] = None):
    """Plots time series linechart for Paralympics Dataset.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the Paralympics data.
        x_col (str): The column to use for the x-axis which is 'year'.
        y_cols (list[str], optional): The column(s) to plot on the y-axis. If None, defaults to 'participants'.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    #default to participants if no y columns provided
    if y_cols is None:
        y_cols = ["participants"]

    #validate columns
    missing_cols = [col for col in [x_col] + y_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: These columns were not found in the DataFrame: {missing_cols}")
        return
    
    #sort by the time column
    df_sorted = df.sort_values(by=x_col)

    #create the line plot
    plt.figure(figsize=(10, 6))
    for y in y_cols:
        plt.plot(df_sorted[x_col], df_sorted[y], marker='o', label=y)

    plt.title("Paralympics Time Series Data")
    plt.xlabel(x_col.capitalize())
    plt.ylabel("Number of Participants")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


#2.8 - identifying values in categorical data
def inspect_categorical_data(df: pd.DataFrame, columns: list[str]):
    """Prints the distinct categorical values and their counts for specified columns.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the Paralympics data.
        columns (list[str]): A list of column names to inspect.

    Returns:
        None
    """

    for col in columns:
        if col not in df.columns:
            print(f"Column '{col}' not found in DataFrame.")
            continue

        print(f"Distinct values in column '{col}':")
        print(df[col].unique())

        print(f"Value counts in column '{col}':")
        print(df[col].value_counts(dropna=False))
   


if __name__ == "__main__":
    # Import required modules
    import pandas as pd
    from pathlib import Path

    # Define project root and data paths
    project_root = Path(__file__).parent.parent.parent
    paralympics_csv_path = project_root / 'activities' / 'data' / 'paralympics_raw.csv'
    paralympics_xlsx_path = project_root / 'activities' / 'data' / 'paralympics_all_raw.xlsx'

    # Load the data
    paralympics_csv_file = pd.read_csv(paralympics_csv_path)
    paralympics_xlsx_file = pd.read_excel(paralympics_xlsx_path, sheet_name=0)

    # Step 1 – Describe Data
    print("CSV data description:")
    describe_dataframe(paralympics_csv_file)

    # Step 2 – Check for Missing Values
    print("\nCSV data quality check:")
    check_dataquality(paralympics_csv_file)

    # Step 3 – Create Histograms
    print("\nCreating histograms for Paralympics CSV data...")
    plot_histogram(paralympics_csv_file, "participants_m")  # or create_histograms(df) if using the other version

    # Step 4 – Create Box Plots
    print("\nCreating box plots for Paralympics CSV data...")
    plot_boxplot(paralympics_csv_file)

    # Optional: create box plots for specific columns only
    # plot_boxplot(paralympics_csv_file, columns=["participants_m", "participants_f"])

    # Step 5 – Create Time Series Linechart
    print("\nCreating time series linechart for Paralympics CSV data...")
    plot_timeseries(paralympics_csv_file, x_col="year", y_cols=["participants", "participants_m", "participants_f"])

    # Step 6 – Inspect Categorical Data
    print("\nInspecting categorical data for Paralympics CSV data...")
    inspect_categorical_data(paralympics_csv_file, columns=["type", "disabilities_included"])

