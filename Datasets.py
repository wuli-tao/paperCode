import os
import random

import cv2
import numpy as np
import pandas as pd
import torch.utils.data as data

import image_utils


# 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'RAFDB.csv'), sep=',', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:, LABEL_COLUMN].values - 1
        # self.valence = dataset.iloc[:, 2].values
        # self.arousal = dataset.iloc[:, 3].values
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            # f = f.split(".")[0]
            # f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'image_data', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __distribute__(self):
        distribute_ = self.label
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # valence = self.valence[idx]
        # arousal = self.arousal[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        # return image, label, idx, valence, arousal
        return image, label, idx


class JAFFEDataSet(data.Dataset):
    def __init__(self, jaffe_path, phase=None, transform=None, basic_aug=False):
        self.transform = transform
        self.jaffe_path = jaffe_path
        self.phase = phase

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        if phase=='test':
            df = pd.read_csv(os.path.join(self.jaffe_path, 'jaffe_test.csv'), sep=',', header=None)
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values
            # self.valence = df.iloc[:, 2].values
            # self.arousal = df.iloc[:, 3].values
            self.file_paths = []
            for f in file_names:
                path = os.path.join(self.jaffe_path, 'jaffe_alignment', f)
                self.file_paths.append(path)
        elif phase=='train':
            df = pd.read_csv(os.path.join(self.jaffe_path, 'jaffe_train.csv'), sep=',', header=None)
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values
            # self.valence = df.iloc[:, 2].values
            # self.arousal = df.iloc[:, 3].values
            self.file_paths = []
            for f in file_names:
                path = os.path.join(self.jaffe_path, 'jaffe_alignment', f)
                self.file_paths.append(path)
        else:
            df = pd.read_csv(os.path.join(self.jaffe_path, 'jaffe.csv'), sep=',', header=None)
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values
            # self.valence = df.iloc[:, 2].values
            # self.arousal = df.iloc[:, 3].values
            self.file_paths = []
            for f in file_names:
                path = os.path.join(self.jaffe_path, 'jaffe_alignment', f)
                self.file_paths.append(path)


        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __distribute__(self):
        distribute_ = self.label
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # valence = self.valence[idx]
        # arousal = self.arousal[idx]
        # augmentation
        if self.transform is not None:
            image = self.transform(image)

        # return image, label, idx, valence, arousal
        return image, label, idx


class FER2013DataSet(data.Dataset):
    def __init__(self, fer2013_path, phase, transform=None, basic_aug=False):
        self.transform = transform
        self.fer2013_path = fer2013_path
        self.phase = phase

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        if phase == 'train':
            df = pd.read_csv(os.path.join(self.fer2013_path, 'train.csv'), header=None)
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values

            self.file_paths = []
            for f in file_names:
                path = os.path.join(self.fer2013_path, 'train', f)
                self.file_paths.append(path)
        else:
            df = pd.read_csv(os.path.join(self.fer2013_path, 'test.csv'), header=None)
            file_names = df.iloc[:, NAME_COLUMN].values
            self.label = df.iloc[:, LABEL_COLUMN].values

            self.file_paths = []
            for f in file_names:
                path = os.path.join(self.fer2013_path, 'test', f)
                self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __distribute__(self):
        distribute_ = self.label
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        # image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class FER2013PlusDataSet(data.Dataset):
    def __init__(self, fer2013plus_path, phase, transform=None, basic_aug=False):
        self.transform = transform
        self.fer2013plus_path = fer2013plus_path
        self.phase = phase
        self.FER2013PlustoLabel = {5: 2, 2: 0, 1: 3, 3: 4, 6: 1, 4: 5, 0: 6}
        self.label = []

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        if phase == 'train':
            df = pd.read_csv(os.path.join(self.fer2013plus_path, 'FER2013Train/train.csv'), header=None)
            file_names = df.iloc[:, NAME_COLUMN].values
            label = df.iloc[:, LABEL_COLUMN].values
            for idx in range(len(label)):
                self.label.append(self.FER2013PlustoLabel[label[idx]])

            self.file_paths = []
            for f in file_names:
                path = os.path.join(self.fer2013plus_path, 'FER2013Train', f)
                self.file_paths.append(path)
        else:
            df = pd.read_csv(os.path.join(self.fer2013plus_path, 'FER2013Test/test.csv'), header=None)
            file_names = df.iloc[:, NAME_COLUMN].values
            label = df.iloc[:, LABEL_COLUMN].values
            for idx in range(len(label)):
                self.label.append(self.FER2013PlustoLabel[label[idx]])

            self.file_paths = []
            for f in file_names:
                path = os.path.join(self.fer2013plus_path, 'FER2013Test', f)
                self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __distribute__(self):
        distribute_ = np.array(self.label)
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class SFEWDataSet(data.Dataset):
    def __init__(self, sfew_path, phase, transform=None, basic_aug=False):
        self.transform = transform
        self.sfew_path = sfew_path
        self.phase = phase
        self.SFEWtoLabel = {5: 4, 2: 1, 1: 2, 3: 3, 6: 0, 4: 6, 0: 5}
        self.label = []

        FILE_COLUMN = 0
        LABEL_COLUMN = 1
        if phase == 'train':
            df = pd.read_csv(os.path.join(self.sfew_path, 'SFEW_2.0_train_NAS_Path.csv'), header=None)
        else:
            df = pd.read_csv(os.path.join(self.sfew_path, 'SFEW_2.0_val_NAS_Path.csv'), header=None)
        self.file_paths = df.iloc[:, FILE_COLUMN].values
        label = df.iloc[:, LABEL_COLUMN].values
        for idx in range(len(label)):
            self.label.append(self.SFEWtoLabel[label[idx]])

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __distribute__(self):
        distribute_ = np.array(self.label)
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

            # if label in [1, 2, 6]:
            #     index = random.randint(0, 4)
            #     image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class ExpWDataSet(data.Dataset):
    def __init__(self, expw_path, phase, transform=None, basic_aug=False):
        self.transform = transform
        self.expw_path = expw_path
        self.phase = phase
        self.ExpWtoLabel = {5: 2, 2: 0, 1: 3, 3: 4, 6: 1, 4: 5, 0: 6}
        self.label = []

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.expw_path, 'expw_modify.csv'), header=None)
        if phase == 'train':
            dataset = df[df[4] == 'train']
        else:
            dataset = df[df[4] == 'test']
        file_names = dataset.iloc[:, NAME_COLUMN].values
        label = dataset.iloc[:, LABEL_COLUMN].values
        for idx in range(len(label)):
            self.label.append(self.ExpWtoLabel[label[idx]])

        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.expw_path, 'ExpwCleaned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __distribute__(self):
        distribute_ = np.array(self.label)
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class AffectNetDataSet(data.Dataset):
    def __init__(self, affectnet_path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        self.affectnet_path = affectnet_path
        self.AffecttoLabel = {5: 2, 2: 4, 1: 3, 3: 0, 6: 5, 4: 1, 0: 6}
        self.label = []

        FILE_COLUMN = 0
        LABEL_COLUMN = 1
        if phase == 'train':
            df = pd.read_csv(
                os.path.join(self.affectnet_path, 'NAS_AffectNetTrain_7_expression_without_contempt_crop.csv'),
                header=None)
        else:
            df = pd.read_csv(
                os.path.join(self.affectnet_path, 'NAS_AffectNetTest_7_expression_without_contempt_crop.csv'),
                header=None)
        self.file_paths = df.iloc[:, FILE_COLUMN].values
        label = df.iloc[:, LABEL_COLUMN].values
        for idx in range(len(label)):
            self.label.append(self.AffecttoLabel[label[idx]])

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __distribute__(self):
        distribute_ = np.array(self.label)
        return (np.sum(distribute_ == 0), np.sum(distribute_ == 1), np.sum(distribute_ == 2), np.sum(distribute_ == 3),
                np.sum(distribute_ == 4), np.sum(distribute_ == 5), np.sum(distribute_ == 6))

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        # augmentation
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

            # if label in [1, 2, 6]:
            #     index = random.randint(0, 4)
            #     image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx
