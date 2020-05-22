'''Simple LSTM example.

Reference: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, cfg):
        super(RNNModel, self).__init__()
        self.cfg = cfg
        in_channels = cfg['in_channels']
        mid_channels = cfg['mid_channels']
        out_channels = cfg['out_channels']
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(True),
            nn.Linear(mid_channels, mid_channels),
        )
        self.rnn = nn.LSTM(mid_channels, mid_channels,
                           cfg['num_layers'], batch_first=True)
        self.decoder = nn.Linear(mid_channels, out_channels)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        hidden = (
            weight.new_zeros(self.cfg['num_layers'],
                             batch_size, self.cfg['mid_channels']),
            weight.new_zeros(self.cfg['num_layers'],
                             batch_size, self.cfg['mid_channels']),
        )
        return hidden

    def forward(self, x, h0, h1):
        out = self.encoder(x)
        out, hidden = self.rnn(out, (h0, h1))
        out = self.decoder(out)
        return out


def rnn_model():
    cfg = {
        'in_channels': 6,
        'mid_channels': 128,
        'out_channels': 2,
        'num_layers': 2,
    }
    return RNNModel(cfg)


def build_model(cfg):
    model = rnn_model()
    # Transfer the model to the crrent GPU device
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device)
    return model


def test():
    model = rnn_model()
    batch = 1
    seq = 8
    in_channels = 6
    x = torch.randn(batch, seq, in_channels)
    hidden = model.init_hidden(batch)
    print(x.shape)
    print(hidden[0].shape)
    print(hidden[1].shape)
    # y = model(x, hidden)
    # print(y.shape)


def to_torch_script():
    model = rnn_model()
    batch = 1
    seq = 8
    in_channels = 6
    x = torch.randn(batch, seq, in_channels)
    hidden = model.init_hidden(batch)
    traced = torch.jit.trace(model, (x, hidden[0], hidden[1]))
    traced.save('./pth/traced.pt')


if __name__ == '__main__':
    # test()
    to_torch_script()
