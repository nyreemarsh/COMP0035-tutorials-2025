# using pathlib for determining path

from pathlib import Path

# this file is src/activities/nyree_code_solutions/tutorial_activities.py
# the other file is src/activities/data/paralymics_raw.csv

project_root = Path(__file__).parent.parent.parent
csv_file = project_root / 'activities' / 'data' / 'paralympics_raw.csv'

print(csv_file.exists())  # this should print True if the path is correct


import csv

if __name__ == '__main__':
    data_file =  csv_file

    with open(data_file) as csv_f:
        csv_reader = csv.reader(csv_f, delimiter=',')
        first_row = next(csv_reader)
        print(first_row)


