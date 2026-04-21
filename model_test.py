import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet

def test_data_process():
    test_data = FashionMNIST(root='./data',
                                train=False,
                                transform=transforms.Compose([transforms.Resize(size=28),transforms.ToTensor()]),
                                download=True)

    test_dataloader = Data.DataLoader(dataset = test_data, batch_size=1, shuffle=True, num_workers=0)

    return  test_dataloader


def test_model_process(model, test_dataloader):
    #设定测试用到的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #将模型放入测试设备中
    model.to(device)
    #初始化参数
    test_correct = 0.0
    test_num = 0
    #只进行前向传播，不计算梯度，从而节省内存
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            #将特征放入测试设备中
            test_data_x = test_data_x.to(device)
            #将标签放入测试设备中
            test_data_y = test_data_y.to(device)
            #设置为评估模式
            model.eval()
            #前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output = model(test_data_x)
            #dim=1，沿着第一维度找max
            pre_lab = torch.argmax(output, dim=1)
            #计算预测正确的数量
            test_correct += torch.sum(pre_lab == test_data_y)
            #累加所有测试样本数
            test_num += test_data_x.size(0)
    #计算测试准确率
    test_acc = test_correct.double().item() / test_num
    print(f"Test Accuracy: {test_acc}")


if __name__ == "__main__":
    #加载模型
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    #加载测试数据
    test_dataloader = test_data_process()
    #加载模型测试的函数
    #test_model_process(model, test_dataloader)

    # 设定测试用到的设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            #设置模型为验证模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = classes[pre_lab.item()]
            label = classes[b_y.item()]
            print(f"预测值: {result}, 真实值: {label} ")