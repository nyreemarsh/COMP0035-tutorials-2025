#2 - data preparation

"""
This module is preparing the Paralympics dataset for use in a dashboard web app for high school students.

The data preparation process includes:
- removing unnecessary columns
- handling missing values
- changing data types
- correcting categorical data
- adding or merging data for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def prepare_paralympics_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the Paralympics dataset for dashboard use.

    Parameters:
        raw_df (pd.DataFrame): The raw Paralympics dataset.

    Returns:
        pd.DataFrame: The cleaned and prepared Paralympics dataset.
    """

    #make a copy of the raw_df
    df = raw_df.copy()

    #remove unnecessary columns
    columns_to_drop = ['highlights', 'URL', 'disabilities_included']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    #rename columns for clarity
    df = df.rename(columns={
        'type': 'season',
        'country': 'host_country',
        'host': 'host_city',
        'start': 'start_date',
        'end': 'end_date',
        'countries': 'countries_attending',
        'events': 'number_of_events',
        'sports': 'number_of_sports',
        'participants_m': 'male_participants',
        'participants_f': 'female_participants',
        'participants': 'total_participants'
    })

    #handle missing values
    numeric_cols = ['countries_attending', 'number_of_events', 'number_of_sports', 'male_participants', 'female_participants', 'total_participants']
    text_cols = ['host_country', 'host_city', 'season']

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[text_cols] = df[text_cols].fillna('Unknown')

    #dropping specific rows with known issues
    df = df.drop(index=[0, 17, 31], errors='ignore')  
    df = df.reset_index(drop=True)

    #handling date values - drop where missing
    if 'year' in df.columns:
        df = df.dropna(subset=['year'])

    if 'start_date' in df.columns:
        df = df.dropna(subset=['start_date'])

    #convert data columns from string to datetime
    if 'start_date' in df.columns:
        df['start_date'] = pd.to_datetime(df['start_date'], format='%d/%m/%Y', errors='coerce')
    if 'end_date' in df.columns:
        df['end_date'] = pd.to_datetime(df['end_date'], format='%d/%m/%Y', errors='coerce')

    #calculate event duration in days
    if all(col in df.columns for col in ['start_date', 'end_date']):
        duration_values = (df['end_date'] - df['start_date']).dt.days.astype('Int64')
        df.insert(df.columns.get_loc('end_date') + 1, 'duration', duration_values)

    #standardise categorical data
    if 'season' in df.columns:
        df['season'] = df['season'].str.strip().str.lower()

    #convert data types
    convert_cols = ['year', 'countries_attending', 'number_of_events', 'number_of_sports', 'male_participants', 'female_participants', 'total_participants']

    for col in convert_cols:
        if col in df.columns:
            df[col] = df[col].astype('Int64')

    #create a combined event label with city + year
    if all(col in df.columns for col in ['host_city', 'year']):
        df['event_label'] = df['host_city'].astype(str) + ' ' + df['year'].astype(str)

    #convert all object columns to string type
    object_cols = df.select_dtypes(include=['object']).columns
    df[object_cols] = df[object_cols].astype('string')

    return df


def merge_npc_codes(paralympics_df: pd.DataFrame, npc_csv_path: Path) -> pd.DataFrame:
    """Merge NPC codes into the cleaned Paralympics dataset.
    
    Parameters:
        paralympics_df (pd.DataFrame): The cleaned Paralympics dataset.
        npc_csv_path (Path): Path to the npc_codes_csv file.
        
    Returns:
        pd.DataFrame: The Paralympics dataset with NPC codes merged.
    """

    #load npc_codes data
    npc_df = pd.read_csv(npc_csv_path, usecols=["Code", "Name"], encoding='utf-8', encoding_errors='ignore')

    #standardise names
    replacement_names = {
        "UK": "Great Britain",
        "USA": "United States of America",
        "Korea": "Republic of Korea",
        "Russia": "Russian Federation",
        "China": "People's Republic of China"
    }
    
    #replace names in host_country column 
    paralympics_df['host_country'] = paralympics_df['host_country'].replace(replacement_names)

    #merge df with left join
    merged_df = paralympics_df.merge(npc_df, left_on='host_country', right_on='Name', how='left')

    #drop duplicate name column
    merged_df = merged_df.drop(columns=['Name'], errors='ignore')

    return merged_df


def validate_prepared_data(df: pd.DataFrame):
    """Visual and printed validation to confirm dataset suitability for dashboard use."""

    print("\nDataframe Info:")

    #1. Where in the world have Paralympic events been held?
    print("Unique Host Countries and City Pairs:")
    host_pairs = df[['host_country', 'host_city']].drop_duplicates().sort_values(by=['host_country'])
    print(host_pairs.to_string(index=False))
    print(f"Total unique host locations: {len(host_pairs)}")

    #2. When have the events been held? (timeline)
    print("\nEvent Timeline:")
    plt.figure(figsize=(10, 6))
    plt.scatter(df['year'], df['host_country'], color='skyblue', s=100, alpha=0.7)
    plt.title('Paralympic Events Timeline')
    plt.xlabel('Year')
    plt.ylabel('Host Country')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    #3. How have the number of sports and events changed over time?
    print("\nNumber of Sports and Events Over Time:")
    plt.figure(figsize=(12, 6))
    plt.plot(df['year'], df['number_of_sports'], marker='o', label='Number of Sports', color='pink')
    plt.plot(df['year'], df['number_of_events'], marker='s', label='Number of Events', color='purple')
    plt.title('Number of Sports and Events Over Time')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    #4. Trends in participant numbers (total, gender, and by season)
    print("\nParticipant Trends Over Time:")
    plt.figure(figsize=(12, 6))
    plt.plot(df['year'], df['total_participants'], label='Total Participants', color='black', linewidth = 2)
    plt.plot(df['year'], df['male_participants'], label='Male Participants', color='blue', linestyle = "--")
    plt.plot(df['year'], df['female_participants'], label='Female Participants', color='magenta', linestyle = "--")
    plt.title('Participant Trends Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Participants')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

    #5. Seasonal Trend Analysis
    print("\nSeasonal Trend Analysis:")
    if 'season' in df.columns:
        plt.figure(figsize=(12, 6))
        for season in df['season'].unique():
            subset = df[df['season'] == season]
            plt.plot(subset['year'], subset['total_participants'], marker='o', label=f'{season.capitalize()} Games')
        plt.title('Seasonal Participant Trends Over Time')
        plt.xlabel('Year')
        plt.ylabel('Number of Participants')
        plt.legend()
        plt.grid(alpha=0.4)
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    paralympics_csv_path = project_root / 'activities' / 'data' / 'paralympics_raw.csv'
    output_path = project_root / 'activities' / 'data' / 'paralympics_cleaned.csv'
    npc_csv_path = project_root / 'activities' / 'data' / 'npc_codes.csv'
    merged_output_path = project_root / 'activities' / 'data' / 'paralympics_with_npc_codes.csv'

    #load the raw data
    raw_paralympics_df = pd.read_csv(paralympics_csv_path)
    cleaned_paralympics_df = prepare_paralympics_data(raw_paralympics_df)

    #merge npc codes
    merged_paralympics_df = merge_npc_codes(cleaned_paralympics_df, npc_csv_path)

    #save the cleaned data
    cleaned_paralympics_df.to_csv(output_path, index=False)
    print(f"Cleaned Paralympics data saved to {output_path}")

    #save the merged data
    merged_paralympics_df.to_csv(merged_output_path, index=False)
    print(f"Merged Paralympics data with NPC codes saved to {merged_output_path}")

    #print(cleaned_paralympics_df.head())

    #print("\nDistinct values in 'season' column:", cleaned_paralympics_df['season'].unique())

    #print(cleaned_paralympics_df[['start_date', 'end_date']].head())
    
    #check for NaN
    print("\nRows with missing NPC codes:")
    print(merged_paralympics_df[merged_paralympics_df['Code'].isna()][['host_country', 'Code']])

    #preview merged data
    print("\nPreview of merged dataframe:")
    print(merged_paralympics_df[['host_country', 'Code']].head(10))

    #print validated data
    print("\nValidating prepared data...")
    validate_prepared_data(merged_paralympics_df)