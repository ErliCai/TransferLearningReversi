import torch as tt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReversiCNN(nn.Module):

    def __init__(self, size, convs, kernel, linear, lr=0.01):
        """
        Create a CNN network for evaluating states in Reversi
        :param size:   size of the Reversi board
        :param convs:  number of channels in each convolutional layer
        :param kernel: kernel size.
        :param linear: linear layer sizes
        """
        super(ReversiCNN, self).__init__()
        convs.insert(0, 2)  # input has three channels
        self.convs = nn.ModuleList([nn.Conv2d(convs[i], convs[i + 1],
                                              kernel_size=kernel, stride=1,
                                              padding=kernel // 2)
                                    for i in range(len(convs) - 1)])

        for _ in convs[1:]:  # calculating the size of the output conv layer
            size = self.cnn_out_size(size, kernel, padding=kernel // 2,
                                     stride=1)
        # linear layers:
        linear.insert(0, convs[-1] * size ** 2)
        linear.append(1)
        self.linear = nn.ModuleList([nn.Linear(linear[i], linear[i + 1])
                                     for i in range(len(linear) - 1)])
        self.sigma = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.parameters(), lr, momentum=0.9)

    @staticmethod
    def cnn_out_size(size, kernel, padding, stride):
        """
        Compute the number of output units in a CNN layer
        """
        return (size - kernel + 2 * padding) // stride + 1

    def forward(self, states):
        for conv in self.convs:
            states = F.relu(conv(states))
        states = states.view(states.shape[0], -1)
        for linear in self.linear[:-1]:
            states = F.relu(linear(states))
        states = self.sigma(self.linear[-1](states).view(-1))
        return states

    def train_on_batch(self, states, estimates):
        """
        Train the network on a batch of examples
        :param states:      4d numpy array (1d array of states as numpy arrays)
        :param estimates:   1d array of probabilities that the player
                            whose turn it is will win the game
        """
        self.optimizer.zero_grad()
        outputs = self(tt.FloatTensor(states))
        loss = self.criterion(outputs, tt.FloatTensor(estimates))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, states):
        """
        Run the network on a batch of states and return predicted values
        :param states: 4d numpy array (1d array of states)
        :return:       1d array of probabilities that the player
                       whose turn it is will win the game
        """
        return self.forward(tt.FloatTensor(states)).detach().numpy()

    def save(self, filename):
        """
        Save the model to disk
        :param filename:
        :return:
        """
        tt.save(self.state_dict(), filename)

    def load(self, filename):
        """
        Load model from disk
        :param filename:
        :return:
        """
        self.load_state_dict(tt.load(filename))