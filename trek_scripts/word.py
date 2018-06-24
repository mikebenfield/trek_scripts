
from torch import nn

class WordRnn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        sz = x.size()
        x = x.view(sz[0], 1, sz[1])
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)
        x = F.log_softmax(x, dim=2)
        sz = s.size()
        return (x.view(sz[0], sz[2]), hidden)

def train(model, loss_f, optimizer, chunk_size, tensors, end_tensor):
    max_len = max(len(tensor) for tensor in tensors)
    batch = torch.zeros([max_len, len(tensors), len(tensors[0, 0])])
    indices = torch.arange(max_len, dtype=torch.long)

    if opts.cuda:
        encoded = encoded.cuda()
        onehot = onehot.cuda()
        indices = indices.cuda()

    for i in range(len(tensors)):
        tensor = tensors[i]
        length = len(tensor)
        batch[0:length, i, :] = tensor
        batch[length:, i, :] = end_tensor
    print('ok')

