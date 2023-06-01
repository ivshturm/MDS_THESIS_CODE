import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

import code_extractor


def preprocess_and_split_csv_input(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # filter 3 top smell
    value_counts = df["Description"].value_counts()
    top_values = value_counts.nlargest(3)
    mask = df["Description"].isin(top_values.index)
    filtered_df = df[mask].copy()

    # form new columns and drop non-useful
    filtered_df["SourceCode"] = filtered_df.apply(lambda row: code_extractor.extract_method_from_line(row['File'],
                                                                                                      row['Line']),
                                                  axis=1)
    filtered_df["SmellLineCode"] = filtered_df.apply(lambda row: code_extractor.extract_line(row['File'], row['Line']),
                                                     axis=1)
    filtered_df = filtered_df[filtered_df['SourceCode'].apply(lambda x: type(x) != tuple)]

    # Convert remaining values in 'SourceCode' column to string
    filtered_df['SourceCode'] = filtered_df['SourceCode'].astype(str)

    filtered_df = filtered_df.drop(['File', 'Line', 'Problem', 'Rule set', 'Rule', 'Package'], axis=1)
    filtered_df = filtered_df.dropna(subset=['SourceCode', 'SmellLineCode'])

    # use .loc to avoid SettingWithCopyWarning
    filtered_df.loc[:, 'Priority'] = filtered_df.loc[:, 'Priority'].astype(str)

    label_map = {label: i for i, label in enumerate(filtered_df['Description'].unique())}
    filtered_df['SmellLabel'] = filtered_df['Description'].map(label_map)

    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(filtered_df['SourceCode'])

    priority_encoder = OneHotEncoder()
    priority_onehot_matrix = priority_encoder.fit_transform(filtered_df['Priority'].values.reshape(-1, 1))

    bow_matrix = hstack([pd.DataFrame(bow_matrix.toarray()), priority_onehot_matrix])

    X_train, X_test, y_train, y_test = train_test_split(bow_matrix, filtered_df['SmellLabel'], test_size=0.2,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test
