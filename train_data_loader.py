from __future__ import print_function, division
import PIL
from torch.utils.data import Dataset


class DF_Dataset(Dataset):
    def __init__(self, df, num_classes, transform=None):
        super(DF_Dataset, self).__init__()
        self.df = df
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sigle_df = self.df.iloc[index]
        image, label = self.get_image(sigle_df)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample["image"] = self.transform(sample['image'])
        return sample

    def get_image(self, sigle_df):
        loc = sigle_df['img_loc']
        image = PIL.Image.open(loc, "r")
        label = int(sigle_df['label'])
        return image, label


class Pseudo_Dataset(Dataset):
    def __init__(self, df, transform=None):
        super(Pseudo_Dataset, self).__init__()
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sigle_df = self.df.iloc[index]
        image = self.get_image(sigle_df,index)
        sample = {"image": image, "index": index}
        if self.transform:
            try:
                sample["image"] = self.transform(sample["image"])
            except Exception:
                print("the image index is {}".format(index))
                print("the image loc is {}".format(sigle_df["img_loc"]))
        return sample

    def get_image(self, sigle_df,index):
        loc = sigle_df['img_loc']
        try:
            image = PIL.Image.open(loc)
        except Exception:
            print("the image loc is {} and index is {}".format(sigle_df["img_loc"],index))
        return image
