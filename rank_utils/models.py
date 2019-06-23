import pprint
from mxnet import nd, gluon
from mxnet.gluon import nn


class HParamsMLP:
    def __init__(self, mlp_layers=[3], mlp_act='softrelu', output_act=None, **kwargs):
        if kwargs:
            print('Unused kwargs')
            print(kwargs)

        self.mlp_layers = mlp_layers
        self.mlp_act = mlp_act
        self.output_act = output_act


class ModelMLP(gluon.Block):
    def __init__(self, hp: HParamsMLP, **kwargs):
        super().__init__(**kwargs)
        self.hp = hp

        print('Hyperparameters:')
        pprint.pprint(self.hp.__dict__, width=1)

        with self.name_scope():
            self.mlp = nn.Sequential(prefix='mlp')
            for width in self.hp.mlp_layers:
                self.mlp.add(nn.Dense(width, flatten=False, activation=self.hp.mlp_act))
                # flatten set to False to only operate on last axis

            # Note: hard to train ranking loss with this activation on, recommend set it to None
            self.output = nn.Dense(1, activation=self.hp.output_act)

    def forward(self, pw_ft=None, **kwargs):
        s = self.mlp(pw_ft)
        s = self.output(s)

        return s


class RankNet(gluon.Block):

    def __init__(self, scorer, loss_type='hinge', **kwargs):
        super().__init__(**kwargs)
        self.scorer = scorer
        self.loss_type = loss_type

    def forward(self, x_i, x_j, t_i, t_j):
        s_i = self.scorer(x_i)
        s_j = self.scorer(x_j)
        s_diff = s_i - s_j
        if self.loss_type == 'hinge':
            loss = nd.relu(1.0 - s_diff * nd.sign(t_i - t_j))
        else:  # more loss_types can be defined here
            loss = nd.sign(t_j - t_i) * s_diff / 2. + nd.log(1 + nd.exp(-s_diff))
        # loss = nd.mean(loss, axis=0)
        return loss
