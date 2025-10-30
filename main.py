import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import Classifier
from dataloader import data_myself


# Train
def train():
    model.train()

    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for id, sample in enumerate(train_data_loader):
            echo_data = sample['data']  # The input shape should be [batch_size, 1, 1400]
            echo_label = sample['label']

            # forward propagation
            outputs = model(echo_data).squeeze()
            # normalization
            mean_out = torch.mean(outputs)
            std_out = torch.std(outputs)
            outputs = (outputs-mean_out) / std_out
            # Labels data type conversion
            echo_label = echo_label.long()
            # 计算loss
            loss = criterion(outputs, echo_label)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculation of epoch indicators
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += echo_label.size(0)
            total_correct += predicted.eq(echo_label).sum().item()

        # Epoch information storage and preservation
        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = total_correct / total_samples
        print('Train Epoch: {} Loss: {:.4f} Accuracy: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        # Save training information
        with open("train_log.txt", 'a') as f:
            f.write('Epoch: {} Loss: {:.4f} Accuracy: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc))


def test():
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predicted = []
    all_targets = []

    for epoch in range(num_epochs):
        with torch.no_grad():
            for id, sample in enumerate(test_data_loader):
                echo_data = sample['data']
                echo_label = sample['label']
                echo_label = echo_label.long()
                # 前向传播
                outputs = model(echo_data).squeeze()
                # 归一化
                mean_out = torch.mean(outputs)
                std_out = torch.std(outputs)
                outputs = (outputs - mean_out) / std_out
                # loss
                loss = criterion(outputs, echo_label)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_samples += echo_label.size(0)
                total_correct += predicted.eq(echo_label).sum().item()
                all_predicted.extend(predicted.cpu().numpy())
                all_targets.extend(echo_label.cpu().numpy())

        epoch_loss = running_loss / len(test_data_loader)
        epoch_acc = total_correct / total_samples

        print('Test Epoch: {} Loss: {:.4f} Accuracy: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        # 保存测试信息
        with open("test_log.txt", 'a') as f:
            f.write('Epoch: {} Loss: {:.4f} Accuracy: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc))

    # 计算混淆矩阵和其他指标（8分类）
    confusion = confusion_matrix(all_targets, all_predicted)
    f1 = f1_score(all_targets, all_predicted, average='macro')
    acc = accuracy_score(all_targets, all_predicted)
    recall = recall_score(all_targets, all_predicted, average='macro')
    precision = precision_score(all_targets, all_predicted, average='macro')

    print('Confusion matrix:')
    print(confusion)
    print('F1 score: {:.4f}'.format(f1))
    print('Accuracy: {:.4f}'.format(acc))
    print('Recall: {:.4f}'.format(recall))
    print('Precision: {:.4f}'.format(precision))
    # 保存结果
    with open("test_result.txt", 'w') as f:
        f.write('Confusion matrix:\n')
        f.write(str(confusion) + '\n')
        f.write('F1 score: {:.4f}\n'.format(f1))
        f.write('Accuracy: {:.4f}\n'.format(acc))
        f.write('Recall: {:.4f}\n'.format(recall))
        f.write('Precision: {:.4f}\n'.format(precision))


if __name__ == '__main__':
    # 定义训练参数
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 300

    # 输入尺寸适应[1,1400]，输出8分类
    model = Classifier(input_size=1, output_size=8)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 dataloader
    train_data_loader = data.DataLoader(
        data_myself(
            data_folder='',
            set='train_data'
        ),
        batch_size=batch_size,
        shuffle=True
    )

    test_data_loader = data.DataLoader(
        data_myself(
            data_folder='',
            set='test_data'
        ),
        batch_size=batch_size,
        shuffle=False
    )
    time_open = time.time()
    train()
    # test()