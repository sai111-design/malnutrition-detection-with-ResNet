import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

class RoboflowMalnutritionDataset(Dataset):
    """Custom dataset for Roboflow malnutrition images"""

    def __init__(self, root_dir, transform=None):
        """Load images and labels"""
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self._load_images()

    def _load_images(self):
        """Load image paths and labels"""
        images_dir = os.path.join(self.root_dir, 'images')
        if os.path.exists(images_dir):
            self._load_flat_structure(images_dir)
        else:
            self._load_class_structure(self.root_dir)

    def _load_flat_structure(self, images_dir):
        """Load from flat images folder"""
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                path = os.path.join(images_dir, filename)

                if 'healthy' in filename.lower():
                    label = 0
                elif 'malnourished' in filename.lower():
                    label = 1
                else:
                    continue

                self.images.append(path)
                self.labels.append(label)

    def _load_class_structure(self, root_dir):
        """Load from class-based folders"""
        class_mapping = {'healthy': 0, 'malnourished': 1}

        for class_name, label in class_mapping.items():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        path = os.path.join(class_dir, filename)
                        self.images.append(path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def get_roboflow_dataloaders(data_dir="data/", batch_size=32, num_workers=4):
    """Get train, val, test dataloaders"""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = RoboflowMalnutritionDataset(
        os.path.join(data_dir, 'train'),
        transform=transform
    )

    val_dataset = RoboflowMalnutritionDataset(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )

    test_dataset = RoboflowMalnutritionDataset(
        os.path.join(data_dir, 'test'),
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    print(f"Test: {len(test_dataset)} images")

    return train_loader, val_loader, test_loader
