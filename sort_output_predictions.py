import pandas as pd
import numpy as np
import sys

filename = sys.argv[1]

def main():
    sample_df = pd.read_csv('sample_submission.csv', encoding='utf-8')
    output_df = pd.read_csv(filename, encoding='utf-8')
    output_df = output_df.replace(np.nan, '', regex=True)
    new_df = sample_df.merge(output_df, left_on='Id', right_on='Id', how='outer')
    new_df = new_df.loc[:, ['Id', 'Predicted_y']]
    new_df.columns = ['Id','Predicted']
    list_parts = filename.split('.'); list_parts[0] = list_parts[0] +'_sorted'
    new_filename = '.'.join(list_parts)
    new_df.to_csv(new_filename, index=False, encoding='utf-8')

if __name__ == '__main__':
    main()
