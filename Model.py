label_targets = [51, 77, 46, 73, 89]
flower_type = ["矮牵牛", "西番莲", "金鱼草", "香睡蓮", "旱金蓮"]
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import os


# 自定义数据集类
class FlowerDataset(Dataset):
    def __init__(self, image_dir, labels_path, target_label, target_size=(224, 224)):
        self.image_dir = image_dir
        mat_data = scipy.io.loadmat(labels_path, squeeze_me=True, struct_as_record=False)
        self.labels = mat_data['labels']
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.binary_labels = (self.labels == target_label).astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = f"image_{idx + 1:05d}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.binary_labels[idx], dtype=torch.float32)
        return image, label


# 定义卷积神经网络模型
class FlowerClassifier(nn.Module):
    def __init__(self):
        super(FlowerClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train_model(target, model_path):
    # 训练参数设置
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化数据集和数据加载器
    dataset = FlowerDataset(image_dir="102flowers\\jpg\\", labels_path='imagelabels.mat', target_label=target)

    # 划分训练集和验证集（8:2比例）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 初始化模型、损失函数和优化器
    model = FlowerClassifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 训练循环
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images).squeeze()
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | "
              f"Train Loss: {epoch_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)

    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


def predict_one_image(image_path, model_path, device='auto'):
    # 设备自动检测
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = FlowerClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 进入评估模式

    # 图像预处理（必须与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载并预处理图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件 {image_path} 不存在")

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    input_tensor = input_tensor.to(device)

    # 执行预测
    with ((torch.no_grad())):
        output = model(input_tensor).squeeze()
        probability = torch.sigmoid(output).item()

    return {
        "output": output,
        "probability": probability,
    }


def set_model(class_num, device='auto'):
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = FlowerClassifier()
    model.load_state_dict(torch.load(f'class{class_num}.pth', map_location=device))
    model.to(device)
    model.eval()
    return model


def vote_method(model_list, image_path, device='auto'):
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vote = [0 for i in range(len(model_list))]
    # 图像预处理（必须与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载并预处理图像
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件 {image_path} 不存在")

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    input_tensor = input_tensor.to(device)

    for i in range(len(model_list)):
        # 执行预测
        with ((torch.no_grad())):
            output = model_list[i](input_tensor).squeeze()
            probability = torch.sigmoid(output).item()
            if probability > 0.5:
                vote[i] = probability
    return vote


def choose_answer(vote):
    if max(vote) == 0:
        return -1
    else:
        return label_targets[vote.index(max(vote))]


def test_with_102flowers():
    model_list = []
    for i in label_targets:
        model_list.append(set_model(i))
    mat_data = scipy.io.loadmat('imagelabels.mat', squeeze_me=True, struct_as_record=False)
    labels = mat_data['labels']
    stat = [0 for k in range(len(label_targets))]
    acc = 0
    for i in range(8189):
        label = labels[i]
        img_name = f"image_{i + 1:05d}.jpg"
        img_path = os.path.join("102flowers\\jpg\\", img_name)
        vote = vote_method(model_list, img_path)
        max_val = max(vote)
        if max_val == 0:
            win = True
            for j in label_targets:
                if j == label:
                    win = False
                else:
                    stat[label_targets.index(j)] += 1
            if win:
                acc += 1
        else:
            if label_targets[vote.index(max_val)] == label:
                acc += 1
            bl = []
            for b in label_targets:
                bl.append(b == label)
            for b in range(len(label_targets)):
                if bl[b] == vote[b] or (bl[b] > 0 and vote[b] > 0):
                    stat[b] += 1
                else:
                    print(f"{b}在{i}出错")
    print(f'投票法在102flowers数据集中的正确率：{acc / 8189}')
    for i in range(len(label_targets)):
        print(f"标签{label_targets[i]}分类器正确率：{stat[i] / 8189}")


def test_other_photos():
    model_list = []
    for i in label_targets:
        model_list.append(set_model(i))
    path = '102flowers\\test\\'
    stat = [0 for i in label_targets]
    acc = 0
    sum = 0
    for i in label_targets:
        this_model=0
        for j in range(5):
            p = path + f'{i}\\00{j+1}.jpg'
            vote = vote_method(model_list, p)
            sum += 1
            if i == choose_answer(vote):
                acc += 1
            for k in range(len(label_targets)):
                if k == label_targets.index(i):
                    if vote[k] > 0:
                        stat[k] += 1
                        this_model+=1
                else:
                    if vote[k] == 0:
                        stat[k] += 1
        print(f'class{i}:在自己类别中的正确次数{this_model}')
    print(f"投票法正确率：{acc / sum}")
    for i in range(len(label_targets)):
        print(f"{flower_type[i]}模型准确率：{stat[i] / sum}")


# 使用示例
if __name__ == "__main__":
    test_with_102flowers()