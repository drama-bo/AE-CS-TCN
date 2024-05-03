import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model import Dynamic_conv1d
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
            # print(id)
            # print(sample)
            echo_data = sample['data']
            echo_label = sample['label']

            # 前向传播
            outputs = model(echo_data).squeeze()
            # 归一化
            mean_out = torch.mean(outputs)
            std_out = torch.std(outputs)
            outputs = (outputs-mean_out) / std_out
            # labels 数据类型转换
            echo_label = echo_label.long()
            # 计算loss
            loss = criterion(outputs, echo_label)
            # print(loss)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # epoch 指标 计算
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += echo_label.size(0)
            total_correct += predicted.eq(echo_label).sum().item()

        # epoch 信息保存以及存储
        epoch_loss = running_loss / len(train_data_loader)
        epoch_acc = total_correct / total_samples
        print('Train Epoch: {} Loss: {:.4f} Accuracy: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
        # 保存训练信息
        with open("train_log.txt", 'a') as f:
            f.write('Epoch: {} Loss: {:.4f} Accuracy: {:.4f}\n'.format(epoch, epoch_loss, epoch_acc))


# def eval():
#     # 在验证集上评估模型
#     model.eval()
#     with torch.no_grad():
#         val_outputs = []
#         val_labels = []
#         for val_data, val_labels_batch in val_loader:
#             val_outputs_batch = model(val_data)
#             val_outputs.append(val_outputs_batch)
#             val_labels.append(val_labels_batch)

#         val_outputs = torch.cat(val_outputs, dim=0)
#         val_labels = torch.cat(val_labels, dim=0)

#         # 计算评估指标
#         val_predictions = torch.argmax(val_outputs, dim=1)
#         val_accuracy = accuracy_score(val_labels, val_predictions)
#         val_precision = precision_score(
#             val_labels, val_predictions, average='weighted')
#         val_recall = recall_score(
#             val_labels, val_predictions, average='weighted')
#         val_f1_score = f1_score(
#             val_labels, val_predictions, average='weighted')
#         val_confusion_matrix = confusion_matrix(val_labels, val_predictions)

#     print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}, '
#           f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1_score:.4f}')


# 在测试集上进行测试
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
                outputs = model(outputs).squeeze()
                # 归一化
                mean_out = torch.mean(outputs)
                std_out = torch.std(outputs)
                outputs = (outputs - mean_out) / std_out
                # labels 数据类型转换
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

    # 计算混淆矩阵和其他指标
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

    model = Dynamic_conv2d(in_planes=64, out_planes=3, kernel_size=3, ratio=0.25, padding=1, K=8, )
    # x = x.to('cuda:0')
    # model.to('cuda')
    # model.attention.cuda()
    # nn.Conv3d()
    model.update_temperature()

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

    # val_data_loader = data.DataLoader(
    #     data_myself(
    #         data_folder='',
    #         set='val_data'
    #     ),
    #     batch_size=batch_size,
    #     shuffle=True
    # )

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