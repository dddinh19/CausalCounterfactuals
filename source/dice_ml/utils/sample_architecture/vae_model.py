import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
    
def preprocess_adj_new(adj):
    adj_normalized = (torch.eye(adj.shape[0]).double() - (adj.transpose(0,1)))
    return adj_normalized

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(torch.eye(adj.shape[0]).double()-adj.transpose(0,1))
    return adj_normalized

class CF_VAE(nn.Module):

    def __init__(self, d, encoded_size, embedding):

        super(CF_VAE, self).__init__()

        self.encoded_size = encoded_size
        self.embedding = embedding
        self.data_size = len(d.ohe_encoded_feature_names)

        self.combiner = CombinerNetwork(num_nodes=self.data_size, encoded_size=self.encoded_size, hidden_dim = 16)
        # self.combiner = CombinerNetwork(input_size=encoded_size*2, hidden_size=50, output_size=encoded_size)

        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = d.get_data_params_for_gradient_dice()

        self.original_data = d.data_df
        df = d.one_hot_encoded_data
        self.encoded_data = df[d.ohe_encoded_feature_names + [d.outcome_name]]

        flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        self.categorical_feature_indexes = d.categorical_feature_indexes
        self.continuous_feature_indexes = d.continuous_feature_indexes
        self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
        self.encoded_start_cat = len(self.encoded_continuous_feature_indexes)



        # Plus 1 to the input encoding size and data size to incorporate the target class label
                # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.encoder_mean = nn.Sequential(
            nn.Linear(self.data_size+1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size)
            )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.data_size+1, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid()
            )

        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size+1, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.data_size),
            nn.Sigmoid()
            )

    def encoder(self, x):
        mean = self.encoder_mean(x)
        logvar = 0.5 + self.encoder_var(x)
        return mean, logvar

    def decoder(self, z):
        mean = self.decoder_mean(z)
        return mean

    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        return torch.sum(-0.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar)), axis=1)

    def forward(self, x, c):
        c = c.view(c.shape[0], 1)
        c = torch.tensor(c).float()
        res = {}
        mc_samples = 50
        em, ev = self.encoder(torch.cat((x, c), 1))
        res['em'] = em
        res['ev'] = ev
        res['z'] = []
        res['x_pred'] = []
        res['mc_samples'] = mc_samples
        for _ in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            combined_vector = self.combiner(z, self.embedding)

            x_pred = self.decoder(torch.cat((combined_vector, c), 1))
            res['z'].append(z)
            res['x_pred'].append(x_pred)

        return res

    def compute_elbo(self, x, c, pred_model):
        c = torch.tensor(c).float()
        c = c.view(c.shape[0], 1)
        em, ev = self.encoder(torch.cat((x, c), 1))

        z = self.sample_latent_code(em, ev)
        combined_vector = self.combiner(z, self.embedding)

        dm = self.decoder(torch.cat((combined_vector, c), 1))

        x_pred = dm
        return x, x_pred, torch.argmax(pred_model(x_pred), dim=1)

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

class CF_VAE2(nn.Module):

    def __init__(self, d, encoded_size, adj_matrix):

        super(CF_VAE2, self).__init__()

        self.encoded_size = encoded_size
        self.adj_matrix = adj_matrix
        # Tìm các hàng có tất cả giá trị bằng 0
        zero_rows = torch.all(adj_matrix == 0, dim=1)

        # Lấy chỉ số của những hàng đó
        self.zero_row_indices = torch.nonzero(zero_rows).view(-1)
        self.adj_matrixforzen = preprocess_adj_new(self.adj_matrix).float()
        self.adj_matrixforzde = preprocess_adj_new1(self.adj_matrix).float()

        self.data_size = len(d.ohe_encoded_feature_names)
        self.Wa = nn.Parameter(torch.zeros(1, self.data_size), requires_grad=True)

        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = d.get_data_params_for_gradient_dice()

        flattened_indexes = [item for sublist in self.encoded_categorical_feature_indexes for item in sublist]
        self.encoded_continuous_feature_indexes = [ix for ix in range(len(self.minx[0])) if ix not in flattened_indexes]
        self.encoded_start_cat = len(self.encoded_continuous_feature_indexes)


        self.encoder_mean = nn.Sequential(
            nn.Linear(self.data_size*2+1, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, self.encoded_size)
            )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.data_size*2+1, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, self.encoded_size),
            nn.Sigmoid()
            )

        # Plus 1 to the input encoding size and data size to incorporate the target class label
        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size+self.data_size+1, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, self.data_size),
            nn.Sigmoid()
            )

    def encoder(self, x):
        mean = self.encoder_mean(x)
        logvar = 0.5 + self.encoder_var(x)
        return mean, logvar
    
    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps
    
    def forward(self, x, c):
        c = c.view(c.shape[0], 1)
        c = torch.tensor(c).float()
        res = {}
        mc_samples = 50
        z = torch.matmul(x, self.adj_matrixforzen)
        em, ev = self.encoder(torch.cat((x, z, c), 1))
        res['em'] = em
        res['ev'] = ev
        res['x_pred'] = []            
        res['x_pred_causal'] = []
        res['mc_samples'] = mc_samples
        res['adj_matrix'] = self.adj_matrix
        res['z'] = z

        for _ in range(mc_samples):
            z_latent = self.sample_latent_code(em, ev)
            x_pred = self.decoder_mean(torch.cat((z_latent, z, c), 1))
            x_pred_causal = torch.matmul(x_pred, self.adj_matrixforzen)
            res['x_pred'].append(x_pred)
            res['x_pred_causal'].append(x_pred_causal)

        return res

    def compute_elbo(self, x, c, pred_model):
        c = torch.tensor(c).float()
        c = c.view(c.shape[0], 1)
        
        z = torch.matmul(x, self.adj_matrixforzen)
        em, ev = self.encoder(torch.cat((x, z, c), 1))

        z_latent = self.sample_latent_code(em, ev)
        x_pred = self.decoder_mean(torch.cat((z_latent, z, c), 1))

        return x, x_pred, torch.argmax(pred_model(x_pred), dim=1)
    
class AutoEncoder(nn.Module):

    def __init__(self, d, encoded_size):

        super(AutoEncoder, self).__init__()

        self.encoded_size = encoded_size
        self.data_size = len(d.encoded_feature_names)
        self.encoded_categorical_feature_indexes = d.get_data_params()[2]

        self.encoded_continuous_feature_indexes = []
        for i in range(self.data_size):
            valid = 1
            for v in self.encoded_categorical_feature_indexes:
                if i in v:
                    valid = 0
            if valid:
                self.encoded_continuous_feature_indexes.append(i)

        self.encoded_start_cat = len(self.encoded_continuous_feature_indexes)

        self.encoder_mean = nn.Sequential(
            nn.Linear(self.data_size, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size)
        )

        self.encoder_var = nn.Sequential(
            nn.Linear(self.data_size, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, self.encoded_size),
            nn.Sigmoid()
         )

        self.decoder_mean = nn.Sequential(
            nn.Linear(self.encoded_size, 12),
            nn.BatchNorm1d(12),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(12, 14),
            nn.BatchNorm1d(14),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(14, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(20, self.data_size),
            nn.Sigmoid()
            )

    def encoder(self, x):
        mean = self.encoder_mean(x)
        logvar = 0.05 + self.encoder_var(x)
        return mean, logvar

    def decoder(self, z):
        mean = self.decoder_mean(z)
        return mean

    def sample_latent_code(self, mean, logvar):
        eps = torch.randn_like(logvar)
        return mean + torch.sqrt(logvar)*eps

    def normal_likelihood(self, x, mean, logvar, raxis=1):
        return torch.sum(-0.5 * ((x - mean)*(1./logvar)*(x-mean) + torch.log(logvar)), axis=1)

    def forward(self, x):
        res = {}
        mc_samples = 50
        em, ev = self.encoder(x)
        res['em'] = em
        res['ev'] = ev
        res['z'] = []
        res['x_pred'] = []
        res['mc_samples'] = mc_samples
    
        for _ in range(mc_samples):
            z = self.sample_latent_code(em, ev)
            x_pred = self.decoder(z)
            res['z'].append(z)
            res['x_pred'].append(x_pred)

        return res

def preprocess_adj_new1(adj):
    adj_normalized = torch.inverse(
        torch.eye(adj.shape[0]).double() - adj.transpose(0, 1)
    )
    return adj_normalized

class MLPEncoder(nn.Module):
    """MLP encoder module."""

    def __init__(
        self,
        n_in,
        n_xdims,
        n_hid,
        n_out,
        adj_A,
        batch_size,
        do_prob=0.0,
        factor=True,
        tol=0.1,
    ):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).double(), requires_grad=True)
        )
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias=True)
        self.fc2 = nn.Linear(n_hid, n_out, bias=True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(
            torch.ones_like(torch.from_numpy(adj_A)).double()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print("nan error \n")

        adj_A1 = torch.sinh(3.0 * self.adj_A)

        adj_Aforz = preprocess_adj_new1(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()

        H1 = F.relu((self.fc1(inputs)))
        x = self.fc2(H1)
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa
    
class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(
        self,
        n_in_node,
        n_in_z,
        n_out,
        encoder,
        data_variable_size,
        batch_size,
        n_hid,
        do_prob=0.0,
    ):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias=True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias=True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(
        self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa
    ):

        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z + Wa) - Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt
    
class CF_MLPAutoEncoder(nn.Module):
    def __init__(self, d, encoded_size, adj_A, batch_size):
        super(CF_MLPAutoEncoder, self).__init__()

        # ... other initializations ...

        self.encoder = MLPEncoder(self.data_size+1, self.data_size, 20, encoded_size, adj_A, batch_size)
        self.decoder = MLPDecoder(encoded_size, encoded_size, self.data_size, self.encoder, self.data_size, batch_size, 20)

    def forward(self, x, c):
        c = c.view(c.shape[0], 1)
        c = torch.tensor(c).float()
        
        x_enc, logits, adj_A1, adj_A, z, z_positive, adj_A_, Wa = self.encoder(torch.cat((x, c), 1))
        mat_z, x_pred, adj_A_tilt = self.decoder(x_enc, z, self.data_size)

        return {
            "x_enc": x_enc,
            "x_pred": x_pred,
            # ... other outputs you may need ...
        }

    def compute_loss(self, x, c, pred_model):
        # Update this function based on your new loss requirements.
        # You might just want a reconstruction loss for an autoencoder.
        c = torch.tensor(c).float()
        c = c.view(c.shape[0], 1)

        outputs = self.forward(x, c)
        x_pred = outputs["x_pred"]

        recon_loss = F.mse_loss(x, x_pred)
        
        return recon_loss  # Replace with your loss calculation
