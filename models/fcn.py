import torch.nn as nn


class Module(nn.Module):
    '''Standard module format.
    '''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self.cuda()
        else:
            raise ValueError
        self.__device = d

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')

    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        else:
            raise NotImplementedError

    @property
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError


class StructureNN(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.
    '''

    def __init__(self):
        super(StructureNN, self).__init__()

    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)


class FNN(StructureNN):
    '''Fully connected neural networks.
    '''

    def __init__(self, ind, outd, layers=2, width=50, activation='relu', initializer='default', outputlayer='None',
                 bias=True):
        super(FNN, self).__init__()
        self.ind = ind
        self.outd = outd
        self.layers = layers
        self.width = width
        self.activation = activation
        self.initializer = initializer
        self.outputlayer = outputlayer
        self.bias = bias

        self.modus = self.__init_modules()
        self.__initialize()

    def forward(self, x):
        for i in range(1, self.layers):
            LinM = self.modus['LinM{}'.format(i)]
            NonM = self.modus['NonM{}'.format(i)]
            x = NonM(LinM(x))

        x = self.modus['LinMout'](x)

        if self.outputlayer == "square":
            x = x ** 2
        elif self.outputlayer == "relu":
            x = F.relu(x)
        elif self.outputlayer == "sigmoid":
            x = F.sigmoid(x)
        elif self.outputlayer == "tanh":
            x = F.tanh(x)

        return x

    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.layers > 1:
            modules['LinM1'] = nn.Linear(self.ind, self.width, bias=self.bias)
            modules['NonM1'] = self.Act
            for i in range(2, self.layers):
                modules['LinM{}'.format(i)] = nn.Linear(self.width, self.width, bias=self.bias)
                modules['NonM{}'.format(i)] = self.Act
            modules['LinMout'] = nn.Linear(self.width, self.outd, bias=self.bias)
        else:
            modules['LinMout'] = nn.Linear(self.ind, self.outd, bias=self.bias)

        return modules

    def __initialize(self):
        for i in range(1, self.layers):
            self.weight_init_(self.modus['LinM{}'.format(i)].weight)
            if self.bias:
                nn.init.constant_(self.modus['LinM{}'.format(i)].bias, 0)

        self.weight_init_(self.modus['LinMout'].weight)
        if self.bias:
            nn.init.constant_(self.modus['LinMout'].bias, 0)

    def update_params(self, theta):

        theta_ind = 0
        for i in range(1, self.layers):
            n_weights = (self.ind if i == 1 else self.width) * self.width
            self.modus['LinM{}'.format(i)].weight = torch.nn.parameter.Parameter(
                torch.from_numpy(theta[theta_ind:theta_ind + n_weights].reshape((self.width, -1)).astype(np.float32)))
            theta_ind += n_weights

            n_biases = self.width
            self.modus['LinM{}'.format(i)].bias = torch.nn.parameter.Parameter(
                torch.from_numpy(theta[theta_ind: theta_ind + n_biases].astype(np.float32)))
            theta_ind += n_biases

        n_weights = self.width * self.outd if self.layers > 1 else self.ind * self.outd
        self.modus['LinMout'].weight = torch.nn.parameter.Parameter(
            torch.from_numpy(theta[theta_ind: theta_ind + n_weights].reshape((self.outd, -1)).astype(np.float32)))
        theta_ind += n_weights

        n_biases = self.outd
        self.modus['LinMout'].bias = torch.nn.parameter.Parameter(
            torch.from_numpy(theta[theta_ind: theta_ind + n_biases].astype(np.float32)))
        theta_ind += n_biases

    def get_params(self):
        N_theta = self.ind * self.width + (self.layers - 2) * self.width ** 2 + self.width * self.outd + (
                    self.layers - 1) * self.width + self.outd if self.layers > 1 else self.ind * self.outd + self.outd
        print(self.width, N_theta)
        theta = np.zeros(N_theta)

        theta_ind = 0
        for i in range(1, self.layers):
            n_weights = (self.ind if i == 1 else self.width) * self.width
            theta[theta_ind:theta_ind + n_weights] = self.modus['LinM{}'.format(i)].weight.detach().numpy().flatten()
            theta_ind += n_weights

            n_biases = self.width
            theta[theta_ind: theta_ind + n_biases] = self.modus['LinM{}'.format(i)].bias.detach().numpy().flatten()
            theta_ind += n_biases

        n_weights = self.width * self.outd if self.layers > 1 else self.ind * self.outd
        theta[theta_ind: theta_ind + n_weights] = self.modus['LinMout'].weight.detach().numpy().flatten()
        theta_ind += n_weights

        n_biases = self.outd
        theta[theta_ind: theta_ind + n_biases] = self.modus['LinMout'].bias.detach().numpy().flatten()
        theta_ind += n_biases

        return theta