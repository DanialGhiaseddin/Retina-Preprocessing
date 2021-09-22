import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(description='DiabeticRetinopathyDetection')
    parser.add_argument('--images_path', default='./datasets/resized_train_cropped/resized_train_cropped')
    parser.add_argument('--csv_path', default='./datasets/trainLabels_cropped.csv')
    parser.add_argument('--output_path', default='./datasets/diabetic_retinopathy_detection')
    parser.add_argument('--standard_folder', choices=['True', 'False'], default='True')
    parser.add_argument('--random_state', default=42)
    parser.add_argument('--test_size', default=0.10)
    parser.add_argument('--histogram_matching', choices=['True', 'False'], default='True')
    parser.add_argument('--reference', default='reference.jpg')
    # parser.add_argument('--n_neighbors', type=int, default=9)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    reference_image_path = os.path.dirname(__file__) + f'/{args.reference}'

    dataframe = pd.read_csv(args.csv_path)

    dataframe['exists'] = dataframe['path'].map(os.path.exists)

    dataframe.dropna(inplace=True)
    dataframe = dataframe[dataframe['exists']]

    files_in_folder = os.listdir(args.images_path)  # dir is your directory path
    files_number = len(files_in_folder)

    if dataframe.shape[0] != files_number:
        print("Warning: Mismatch between csv file and files in folder")
        print("CSV: ", dataframe.shape[0], "### Folder: ", files_number)

    x_train, x_test, y_train, y_test = train_test_split(dataframe.index.values,
                                                        dataframe.level.values,
                                                        test_size=args.test_size,
                                                        random_state=args.random_state,
                                                        stratify=dataframe.level.values)

    dataframe['data_type'] = ['not_set'] * dataframe.shape[0]
    dataframe.loc[x_train, 'data_type'] = 'train'
    dataframe.loc[x_test, 'data_type'] = 'test'
