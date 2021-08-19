"# -- coding: UTF-8 --"
import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, height, width):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.height=height
        self.width=width

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wci = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wcf = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))
        self.Wco = nn.Parameter(torch.zeros(int(1), int(self.hidden_channels), int(self.height), int(self.width)))


    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1], height=32, width=32):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self.height = height
        self.width = width
        self._all_layers = []
        self.dropout = nn.Dropout(0.7)
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, self.height, self.width)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, seq, input):
        internal_state = []
        outputs = []
        xout = Variable(torch.zeros(int(input.size()[0]), int(input.size()[1]), int(self.hidden_channels[0]), int(self.height), int(self.width))).cuda()
        for step in range(self.step):
            # a = input[1]
            if step < 2:
                x = seq(input[step])
                xout[step] = x
            else:
                x = xout[0]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
                x = self.dropout(x)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, xout


if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=128, hidden_channels=[128, 128, 128, 128, 128], kernel_size=3, step=2,
                        effective_step=[1], height=32, width=32).cuda()
    loss_fn = torch.nn.MSELoss()

    input = Variable(torch.randn(2, 64, 128, 32, 32)).cuda()
    # target = Variable(torch.randn(2, 64, 128, 32, 32)).double().cuda()

    output = convlstm(input)
    output = output[0][0].double()
    print(output.size())
    # res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    # print(res)