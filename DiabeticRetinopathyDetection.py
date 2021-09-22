import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from skimage.exposure import match_histograms
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='DiabeticRetinopathyDetection')
    parser.add_argument('--images_path', default='./datasets/resized_train_cropped/resized_train_cropped')
    parser.add_argument('--csv_path', default='./datasets/trainLabels_cropped.csv')
    parser.add_argument('--output_path', default='./datasets/diabetic_retinopathy_detection')
    parser.add_argument('--standard_folder', choices=['True', 'False'], default='True')
    parser.add_argument('--random_state', default=42)
    parser.add_argument('--test_size', default=0.10)
    parser.add_argument('--histogram_matching', choices=['True', 'False'], default='False')
    parser.add_argument('--histogram_threshold', default=1)
    parser.add_argument('--reference', default='reference.jpg')
    parser.add_argument('--normal_limit', default=700)
    # parser.add_argument('--n_neighbors', type=int, default=9)
    return parser.parse_args()


def histogram_matching(reference_img, target, threshold=50):
    reference_img = np.array(reference_img)
    target = np.array(target)

    his_matched = match_histograms(target, reference_img, multichannel=True)
    mask = np.mean(target, axis=2)

    mask[mask < threshold] = 0.0
    mask[mask >= threshold] = 1.0

    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)

    his_matched = his_matched * mask
    his_matched = his_matched.astype(np.uint8)

    return his_matched, mask


if __name__ == '__main__':
    args = get_args()
    reference_image_path = os.path.dirname(__file__) + f'/{args.reference}'

    if args.histogram_matching == 'True':
        reference = Image.open(args.reference)
        reference = np.array(reference)
    else:
        reference = None

    dataframe = pd.read_csv(args.csv_path)

    dataframe['path'] = dataframe['image'].map(lambda x: os.path.join(args.images_path, '{}.jpeg'.format(x)))

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

    labels = dataframe['level'].unique()

    print(f"Total {dataframe.shape[0]} data exist in folder")

    Path(f"{args.output_path}/train").mkdir(parents=True, exist_ok=True)
    Path(f"{args.output_path}/test").mkdir(parents=True, exist_ok=True)

    if args.standard_folder == 'True':
        for label in labels:
            path_train = os.path.join(f"{args.output_path}/train/", str(label))
            path_test = os.path.join(f"{args.output_path}/test/", str(label))
            os.mkdir(path_train)
            os.mkdir(path_test)

    counter = 0
    for index, row in tqdm(dataframe.iterrows()):
        src_path = row['path']
        file_name = src_path.split('/')[-1]
        if args.standard_folder == 'True':
            dest_path = f"{args.output_path}/{row['data_type']}/{row['level']}/{file_name}"
        else:
            dest_path = f"{args.output_path}/{row['data_type']}/{file_name}"

        image = Image.open(src_path)
        if args.histogram_matching == 'True':
            image = np.array(image)
            matched, masked = histogram_matching(reference, image, args.histogram_threshold)
            matched_image = Image.fromarray(matched.astype('uint8'), 'RGB')
        else:
            matched_image = image

        if row['data_type'] == 'train' and int(row['level']) == 0:
            if counter < args.normal_limit:
                matched_image.save(dest_path)
                counter += 1
        else:
            matched_image.save(dest_path)
