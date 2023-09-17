import torch
import torch.nn as nn


# Define a simple GRU-based model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size

        # GRU layer for processing x_t
        self.gru_t = nn.GRU(input_size, hidden_size, batch_first=True)

        # GRU layer for processing x_t+1
        self.gru_t1 = nn.GRU(input_size, hidden_size, batch_first=True)

        # Linear layer to map hidden state to output
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x_t, x_t1):
        # Forward pass for x_t
        h_t, _ = self.gru_t(x_t)
        y_t_pred = self.linear(h_t)

        # Forward pass for x_t+1 using h_t from x_t
        h_t1, _ = self.gru_t1(x_t1, h_t)
        y_t1_pred = self.linear(h_t1)

        return y_t_pred, y_t1_pred


# Example usage:
input_size = 10  # Change this to match your input feature size
hidden_size = 32  # Adjust the hidden size as needed
output_size = 1  # Change this to match your output size

# Create an instance of the model
model = GRUModel(input_size, hidden_size, output_size)

# Define a loss function (e.g., Mean Squared Error) and an optimizer (e.g., Adam)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
