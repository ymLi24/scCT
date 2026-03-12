import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

DIM = 2000

class Encoder(nn.Module):
    """
    Encoder architecture in PyTorch.

    Maps the data into the latent space.
    """
    def __init__(self, latent_dim, input_dim=DIM, do_spectral_norm=False):
        super(Encoder, self).__init__()
        
        # Input dimension includes dataset one-hot encoding size
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.layer_1 = nn.Linear(self.input_dim, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)
        
        if do_spectral_norm:
            self.layer_1 = spectral_norm(self.layer_1)
            self.layer_2 = spectral_norm(self.layer_2)
            self.layer_3 = spectral_norm(self.layer_3)
        
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.xavier_uniform_(self.layer_3.weight)

    def forward(self, x):
        """
        Forward pass for the encoder.

        Args:
            x (Tensor): Input expression data.
        """
        x = self.leaky_relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_3(x)
        return x

    
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim=DIM):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.layer_1 = nn.Linear(self.latent_dim, 1024)
        self.layer_2 = nn.Linear(1024, output_dim*2)
        self.layer_3 = nn.Linear(output_dim*2, output_dim)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.2)  # 稀疏性增强
        
        # Initialize weights
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.xavier_uniform_(self.layer_3.weight)

    def forward(self, z):
        z = self.leaky_relu(self.layer_1(z))
        z = self.dropout(z) 
        z = self.leaky_relu(self.layer_2(z))
        z = self.dropout(z) 
        z = self.layer_3(z)
        # z = self.relu(z)
        return z
    
class Discriminatorxz(nn.Module):
    """
    Discriminator architecture in PyTorch.

    Discriminates between pairs (E(x), x) and (z, G(z)).
    """
    def __init__(self, z_dim, x_dim=DIM, do_spectral_norm=False):
        super(Discriminatorxz, self).__init__()
        # D(x) branch
        self.x_layer_1 = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        # D(z) branch
        self.z_layer_1 = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        # Combined layers
        self.y_layer_1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.y_layer_2 = nn.Linear(128, 1)
        
        if do_spectral_norm:
            self.x_layer_1[0] = spectral_norm(self.x_layer_1[0])
            self.z_layer_1[0] = spectral_norm(self.z_layer_1[0])
            self.y_layer_1[0] = spectral_norm(self.y_layer_1[0])
            self.y_layer_2 = spectral_norm(self.y_layer_2)
        
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.x_layer_1[0].weight)
        nn.init.xavier_uniform_(self.z_layer_1[0].weight)
        nn.init.xavier_uniform_(self.y_layer_1[0].weight)
        nn.init.xavier_uniform_(self.y_layer_2.weight)

    def forward(self, x, z):
        x_out = self.x_layer_1(x)
        z_out = self.z_layer_1(z)
        y = torch.cat([x_out, z_out], dim=1)
        y = self.y_layer_1(y)
        intermediate = y
        logits = self.y_layer_2(y)
        # logits = torch.sigmoid(logits)
        return logits, intermediate



class Discriminatorxx(nn.Module):
    """
    Discriminator architecture in PyTorch.

    Discriminates between (x, x) and (x, rec_x).
    """
    def __init__(self, input_dim = DIM, do_spectral_norm=False):
        super(Discriminatorxx, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(2 * input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.layer_2 = nn.Linear(128, 1)
        
        if do_spectral_norm:
            self.layer_1[0] = spectral_norm(self.layer_1[0])
            self.layer_2 = spectral_norm(self.layer_2)
        
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.layer_1[0].weight)
        nn.init.xavier_uniform_(self.layer_2.weight)

    def forward(self, x, rec_x):
        net = torch.cat([x, rec_x], dim=1)
        net = self.layer_1(net)
        intermediate = net
        logits = self.layer_2(net)
        # logits = torch.sigmoid(logits)
        return logits, intermediate

class Discriminatorzz(nn.Module):
    """
    Discriminator architecture in PyTorch.

    Discriminates between (z, z) and (z, rec_z).
    """
    def __init__(self, input_dim, do_spectral_norm=False):
        super(Discriminatorzz, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(2 * input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.layer_2 = nn.Linear(32, 1)
        
        if do_spectral_norm:
            self.layer_1[0] = spectral_norm(self.layer_1[0])
            self.layer_2 = spectral_norm(self.layer_2)
        
        # Initialize weights using Xavier uniform
        nn.init.xavier_uniform_(self.layer_1[0].weight)
        nn.init.xavier_uniform_(self.layer_2.weight)

    def forward(self, z, rec_z):
        net = torch.cat([z, rec_z], dim=1)
        net = self.layer_1(net)
        intermediate = net
        logits = self.layer_2(net)
        # logits = torch.sigmoid(logits)
        return logits, intermediate


class BatchClassifier(nn.Module):
    def __init__(self, latent_dim, num_batches):
    # def __init__(self, latent_dim, num_batches, do_spectral_norm=False):
        super(BatchClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(2048, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(256, num_batches)

        # if do_spectral_norm:
        #     self.fc1 = spectral_norm(self.fc1)
        #     self.fc2 = spectral_norm(self.fc2)
        #     self.fc3 = spectral_norm(self.fc3)
        #     self.fc_out = spectral_norm(self.fc_out)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, z):
        x = self.fc1(z)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)

        logits = self.fc_out(x)
        return logits


class CellClassifier(nn.Module):
    def __init__(self, latent_dim, num_celltypes):
    # def __init__(self, latent_dim, num_celltypes, do_spectral_norm=False):
        super(CellClassifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(2048, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(256, num_celltypes)

        # if do_spectral_norm:
        #     self.fc1 = spectral_norm(self.fc1)
        #     self.fc2 = spectral_norm(self.fc2)
        #     self.fc3 = spectral_norm(self.fc3)
        #     self.fc_out = spectral_norm(self.fc_out)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

    def forward(self, z):
        x = self.fc1(z)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.drop3(x)

        logits = self.fc_out(x)
        return logits

