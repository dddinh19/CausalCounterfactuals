from torch import nn, sigmoid
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

class FFNetwork(nn.Module):
    def __init__(self, input_size, is_classifier=True):
        super(FFNetwork, self).__init__()
        self.is_classifier = is_classifier
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        output = nn.functional.softmax(logits, dim=1)
        if not self.is_classifier:
            out = 3 * out  # output between 0 and 3
        return out

class DirectedWeightedGraphEmbedding(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, encoded_size):
        super(DirectedWeightedGraphEmbedding, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, encoded_size)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # Lưu ý rằng chúng tôi đang sử dụng edge_weight trong quá trình convolution
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        # Lớp fully connected để thu được kích thước nhúng mong muốn
        x = self.fc(x)

        return x

    
class MulticlassNetwork(nn.Module):
    def __init__(self, input_size: int, num_class: int):
        super(MulticlassNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, num_class)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear_relu_stack(x)
        out = self.softmax(x)

        return out
    
class AdvancedNet(nn.Module):
    def __init__(self, x_train):
        super(AdvancedNet, self).__init__()
        # Define your layers here
        self.layer1 = nn.Linear(x_train.shape[1], 32)
        self.layer2 = nn.BatchNorm1d(32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Dropout(0.5) # Regularization
        self.output_layer = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x) # Batch normalization
        x = F.relu(self.layer3(x))
        x = self.layer4(x) # Dropout
        x = torch.softmax(self.output_layer(x), dim=1)
        return x
    
def train_and_save_model(train_data, x_train, model_path):
   
    batch_size = 500
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = AdvancedNet(x_train)  # Initialize your model architecture
    # We will use Adam as our optimizer
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.float()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_path)

def valid_model(model, val_dataset):
    valid_loader = DataLoader(val_dataset, batch_size=25)
    criterion = nn.CrossEntropyLoss() 

    valid_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss()  # Hoặc hàm loss phù hợp với bài toán của bạn

    with torch.no_grad():
        for data, target in valid_loader:
            data = data.float()
            output = model(data)
            valid_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = 100. * correct / len(valid_loader.dataset)
    print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%')
