import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np

from src.utils import accuracy_fn, macrof1_fn

## MS2
class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_dim=512, num_layers=5, activation_fn='relu', use_bias=True):
        """
        Initialize the network.
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            hidden_dim (int): size of hidden layers (default: 512)
            num_layers (int): number of hidden layers (default: 5)
            activation_fn (str): activation function name (default: 'relu', other choices: 'sigmoid', 'tanh', 'elu')
            use_bias (bool): whether to use bias in linear layers (default: True)
            
        """
        super().__init__()

        self.activation_fn = activation_fn

        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_dim, bias=use_bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(p = 0.2)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=use_bias) for m in range(num_layers - 1)])
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, n_classes, bias=use_bias)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = self.dropout(self.apply_activation(self.fc1(x)))
        
        for layer in self.hidden_layers:
            x = self.apply_activation(layer(x))
            x = self.dropout(x)
            
        preds = self.fc_out(x)
        return preds

    def apply_activation(self, x):
        """
        Apply the specified activation function to the input tensor.

        Arguments:
            x (tensor): input tensor
        Returns:
            x (tensor): output tensor after applying the activation function
        """
        if self.activation_fn == 'relu':
            return F.relu(x)
        elif self.activation_fn == 'sigmoid':
            return torch.sigmoid(x)
        elif self.activation_fn == 'tanh':
            return torch.tanh(x)
        elif self.activation_fn == 'elu':
            return F.elu(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, filters = (32, 64, 128), dropouts = (0.3, 0.4, 0.25)):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=filters[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[2], kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout1 = nn.Dropout(dropouts[0])
        self.dropout2 = nn.Dropout(dropouts[1])
        self.dropout3 = nn.Dropout(dropouts[2])
        
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.bn2 = nn.BatchNorm2d(filters[2])
        self.bn3 = nn.BatchNorm1d(filters[2] * 3 * 3)
        
        self.flatten = nn.Flatten()
        
        # Calculate the input size to the first fully connected layer
        self.fc1 = nn.Linear(filters[2] * 3 * 3, 512)  # Adjusted based on 28x28 input with 3 pooling layers
        self.fc2 = nn.Linear(512, n_classes)
        
        self._initialize_weights()

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.bn3(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout3(x)

        return self.fc2(x)   
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, dropout=0.2):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # MHSA + residual connection.
        out = x + self.dropout1(self.mhsa(self.norm1(x)))
        # Feedforward + residual connection
        out = out + self.mlp(self.norm2(out))
        return out

class MyMSA(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super(MyMSA, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        assert hidden_d % n_heads == 0, f"Can't divide dimension {hidden_d} into {n_heads} heads"
        self.head_d = hidden_d // n_heads

        self.qkv = nn.Linear(hidden_d, 3 * hidden_d)
        self.softmax = nn.Softmax(dim=-1)

        self.out_projection = nn.Linear(hidden_d, hidden_d)

    def forward(self, x):
        batch_size, seq_len, d = x.shape
        assert d == self.hidden_d, f"Input dimension {d} does not match model dimension {self.hidden_d}"

        # Apply qkv linear layer and split the results
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_d)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_d)
        attention = self.softmax(scores)
        context = torch.matmul(attention, v)

        # Concatenate heads and apply output projection
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_d)
        out = self.out_projection(context)

        return out

class MyViT(nn.Module):
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches=4, n_blocks=8, hidden_d=64, n_heads=4, out_d=10,
                 position_type="learnable", dropout=0.2):
        """
        Initialize the network.
        """
        super().__init__()

        self.chw = chw # (C, H, W)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0 # Input shape must be divisible by number of patches
        assert chw[2] % n_patches == 0
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # Positional embedding
        self.positional_embeddings = self.get_positional_embeddings(n_patches ** 2 + 1, hidden_d, position_type)

        # Layer Normalization
        self.pre_norm = nn.LayerNorm(hidden_d)

        # Transformer blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads, dropout=dropout) for _ in range(n_blocks)])

        # Layer Normalization
        self.post_norm = nn.LayerNorm(hidden_d)

        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, hidden_d),
            nn.Tanh(),
            nn.Linear(hidden_d, out_d)
        )

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        n = x.shape[0]

        # Divide images into patches.
        patches = self.patchify(x, self.n_patches)

        # Map the vector corresponding to each patch to the hidden size dimension.
        tokens = self.linear_mapper(patches)

        # Add classification token to the tokens.
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Add positional embedding.
        positional_embeddings = self.positional_embeddings.repeat(n, 1, 1)
        out = tokens + positional_embeddings

        out = self.pre_norm(out)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Get the classification token only & Layer normalization after the last block
        preds = self.post_norm(out[:, 0, :])

        # Map to the output distribution.
        preds = self.mlp(preds)

        return preds

    def get_positional_embeddings(self, sequence_length, d, position_type):
        if (position_type == "learnable"):
            pos_embed = nn.Parameter(torch.randn(sequence_length, d), requires_grad=True)
        elif (position_type == "trigonometric"):
            pos_embed = torch.zeros(sequence_length, d)
            pos = torch.arange(sequence_length).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            
            pos_embed[:, 0::2] = torch.sin(pos * div_term)
            pos_embed[:, 1::2] = torch.cos(pos * div_term)
        else:
            raise ValueError
        return pos_embed

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        # Reshape x to (n, c, h, w)
        x_reshaped = images.view(n, c, h, w)

        for idx, image in enumerate(x_reshaped):
            for i in range(n_patches):
                for j in range(n_patches):
                    # Extract the patch of the image.
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    # Flatten the patch and store it.
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, verbose=False):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)


    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader, ep)

            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch
             
    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """

        self.model.train()
        total_loss = 0

        y_true_train = []
        y_pred_train = []        
        for it, batch in enumerate(dataloader):
            x, y = batch
            logits = self.model(x)
            loss = self.criterion(logits, y.long())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            y_pred_train.extend(logits.detach().argmax(dim=-1).tolist())
            y_true_train.extend(y.detach().tolist())

            total_loss += loss
            if (self.verbose):
                print('\rEp {}/{}, it {}/{}: loss train: {:.2f}, '
                      .format(ep + 1, self.epochs, it + 1, len(dataloader), loss), end='')

        if (self.verbose):
            print(f"average loss: {total_loss/it:.2f}")

            total_correct = len([True for x, y in zip(y_pred_train, y_true_train) if x==y])
            total = len(y_pred_train)
            accuracy = total_correct * 100 / total
                
            print(f"          Train Accuracy(%): {accuracy:.4f} == {total_correct}/{total}")
            print("-------------------------------------------------------------")

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.model.eval()
        with torch.no_grad():
            pred_labels = []
            for it, batch in enumerate(dataloader):
                # Get batch of data.
                x = batch[0] if isinstance(batch, (list, tuple)) else batch

                # Get predicted labels
                preds = torch.argmax(self.model(x), dim=1)
                pred_labels.append(preds)
        pred_labels = torch.cat(pred_labels, dim=0)
        return pred_labels
    
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
