import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN2DModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(in_channels=hidden_dim, out_channels=num_class, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm2d(input_dim)
        self.bn2=nn.BatchNorm2d(hidden_dim)

    def forward(self, x):
        (h,w,c)=x.shape
        x=torch.unsqueeze(x.permute([2, 0, 1]), 0)
        x=self.bn1(x)
        x = F.relu(self.conv1(x))
        x=self.bn2(x)
        x = F.relu(self.conv2(x))
        x = torch.squeeze(x, 0).permute([1, 2, 0]).reshape([h * w, -1])
        # x = x.view(x.size(0), -1)
        # x = torch.relu(self.fc1(x))
        # x = self.fc2(x)
        x = F.softmax(x, -1)
        return x