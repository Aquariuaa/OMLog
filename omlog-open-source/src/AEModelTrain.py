import pandas as pd
import time
import torch
from dataset import LogDataset
from partition import partition
from AEfeature import extractFeatures
from torch.utils.data import DataLoader

dataset = 'BGL2'
parsed_path = '../data/{}.log_structured.csv'.format(dataset, dataset)
parsed_log_df = pd.read_csv(parsed_path)
session_train, session_test = partition(parsed_log_df, 'timestamp', 100, False, 0.8)
num_components, num_events, num_levels, uniq_events, session_train, session_test = extractFeatures(session_train, session_test)
num_classes = num_events
print(f'Number of componens: {num_components}, number of training events: {num_classes}, number of levels: {num_levels}.')
batch_size = 1024
eval_batch_size = 1024
step_size = 1
window_size = 10

dataset_train = LogDataset(session_train,
                           window_size,
                           step_size,
                           num_classes)

dataset_test = LogDataset(session_test,
                          window_size,
                          step_size,
                          num_classes)


def collate_fn(batch_input_dict):
    keys = [input_dict['session_key'] for input_dict in batch_input_dict]
    templates = [input_dict['templates'] for input_dict in batch_input_dict]
    event_ids = [input_dict['eventids'] for input_dict in batch_input_dict]
    elapsed_time = [input_dict['elapsedtime'] for input_dict in batch_input_dict]
    components = [input_dict['components'] for input_dict in batch_input_dict]
    levels = [input_dict['levels'] for input_dict in batch_input_dict]

    next_logs = [input_dict['next'] for input_dict in batch_input_dict]
    anomaly = [input_dict['anomaly'] for input_dict in batch_input_dict]
    autoencoder_pred = [input_dict['autoencoder_pred'] for input_dict in batch_input_dict]

    return {'session_key': keys,
            'templates': templates,
            'eventids': event_ids,
            'elapsedtime': elapsed_time,
            'components': components,
            'levels': levels,
            'next': next_logs,
            'anomaly': anomaly,
            'autoencoder_pred': autoencoder_pred}


dataloader_train = DataLoader(dataset_train,
                              collate_fn=collate_fn,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=False)

dataloader_test = DataLoader(dataset_test,
                             collate_fn=collate_fn,
                             batch_size=eval_batch_size,
                             shuffle=False,
                             pin_memory=False)


class AutoEncoderEmbedding(torch.nn.Module):
    def __init__(self, num_components, num_levels):
        super(AutoEncoderEmbedding, self).__init__()
        self.num_components = num_components
        self.num_levels = num_levels

        components_embedding = torch.vstack([torch.eye(num_components), torch.zeros(1, num_components)])
        levels_embedding = torch.vstack([torch.eye(num_levels), torch.zeros(1, num_levels)])
        self.component_embedder = torch.nn.Embedding.from_pretrained(components_embedding, freeze=True)
        self.level_embedder = torch.nn.Embedding.from_pretrained(levels_embedding, freeze=True)

    def forward(self, input_dict):
        components = torch.tensor(input_dict['components'])
        levels = torch.tensor(input_dict['levels'])
        time_elapsed = torch.tensor(input_dict['elapsedtime']).unsqueeze(-1).cuda()

        components[self.num_components < components] = self.num_components
        levels[self.num_levels < levels] = self.num_levels

        components_embedding = self.component_embedder(components.cuda())
        levels_embedding = self.level_embedder(levels.cuda())
        return torch.cat([time_elapsed, components_embedding, levels_embedding], dim=2)


class AutoEncoder(torch.nn.Module):
    def __init__(self,
                 num_components,
                 num_levels,
                 window_size):
        super(AutoEncoder, self).__init__()
        self.EmbeddingLayer = AutoEncoderEmbedding(num_components, num_levels)

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(window_size * (num_components + num_levels + 1), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 3)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 12),
            torch.nn.ReLU(),
            torch.nn.Linear(12, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, window_size * (num_components + num_levels + 1)),
            torch.nn.Tanh()
        )

    def forward(self, input_dict):
        embedding_matrix = self.EmbeddingLayer(input_dict)
        embedding = embedding_matrix.view(embedding_matrix.size(0), -1)
        encoding = self.encoder(embedding)
        return embedding, self.decoder(encoding)


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self,
                 num_components,
                 num_levels,
                 window_size):
        super(VariationalAutoEncoder, self).__init__()
        self.EmbeddingLayer = AutoEncoderEmbedding(num_components, num_levels)
        self.fc1 = torch.nn.Linear(window_size * (num_components + num_levels + 1), 100)
        self.fc21 = torch.nn.Linear(100, 10)
        self.fc22 = torch.nn.Linear(100, 10)
        self.fc3 = torch.nn.Linear(10, 100)
        self.fc4 = torch.nn.Linear(100, window_size * (num_components + num_levels + 1))

    def encode(self, embedding):
        h1 = torch.nn.functional.relu(self.fc1(embedding))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = torch.autograd.Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = torch.nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, input_dict):
        embedding_matrix = self.EmbeddingLayer(input_dict)
        embedding = embedding_matrix.view(embedding_matrix.size(0), -1)

        mu, logvar = self.encode(embedding)
        z = self.reparametrize(mu, logvar)
        return embedding, self.decode(z), mu, logvar


reconstruction_function = torch.nn.MSELoss(reduction='sum')


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD


batch_cnt = 0
learning_rate = 2e-3
num_epochs = 20
total_loss = 0
thresh = 0.02
# training_losses = []

model = AutoEncoder(num_components, num_levels, window_size + 1).cuda()
# model.load_state_dict(torch.load('../checkpoint/bgl_ae_08_epoch10.pth'))
criterion = torch.nn.MSELoss()
# model = VariationalAutoEncoder(num_components, num_levels, window_size+1).cuda()
# criterion = loss_function

eval_criterion = torch.nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

start_time = time.time()
for epoch in range(num_epochs):
    model.train()

    for batch in dataloader_train:
        batch_cnt += 1

        batch_embedding, output = model(batch)
        batch_loss = criterion(output, batch_embedding)
        # batch_embedding, output, mu, logvar = model(batch)
        # batch_loss = criterion(output, batch_embedding, mu, logvar)

        total_loss += batch_loss.mean()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    print(f'[{epoch + 1}|{num_epochs}] Training finished, training loss: {total_loss / batch_cnt :.3f}.')

torch.save(model.state_dict(), '../model/unilog_autoencoder.pth')

end_time = time.time()
print("AE training time:", end_time-start_time)
# 20 epoch-1558