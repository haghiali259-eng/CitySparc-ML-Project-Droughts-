# this program tries to use weather data to predict drought levels (PDSI)
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# load the data

avgtemp = pd.read_csv("data avgtemp.csv", comment="#")
maxtemp = pd.read_csv("data maxtemp.csv", comment="#")
precip = pd.read_csv("data precipitation.csv", comment="#")
pdsi = pd.read_csv("data PDSI.csv", comment="#")
phdi = pd.read_csv("data PHDI.csv", comment="#")

# concatenate data

data = avgtemp.merge(maxtemp, on="Date")
data = data.merge(precip, on="Date")
data = data.merge(pdsi, on="Date")
data = data.merge(phdi, on="Date")

# turn Date into a proper date format
data["Date"] = pd.to_datetime(data["Date"], format="%Y%m")

# Put data in time order
data = data.sort_values("Date")
data = data.reset_index(drop=True)

# select inputs and outputs

# these are the inputs
features = ["avtemp", "maxtemp", "precipitation"]

# target variable
target = "PDSI"

# turn into arrays
X = data[features].values
y = data[target].values.reshape(-1, 1)

# split into training and validation sets (80-20 split)

split = int(len(X) * 0.8)

X_train = torch.tensor(X[:split], dtype=torch.float32)
y_train = torch.tensor(y[:split], dtype=torch.float32)

X_val = torch.tensor(X[split:], dtype=torch.float32)
y_val = torch.tensor(y[split:], dtype=torch.float32)

print("Train size:", X_train.shape)
print("Val size:", X_val.shape)

# make neural network

class MyNet(nn.Module):
    def __init__(self, inputs):
        super(MyNet, self).__init__()
        self.l1 = nn.Linear(inputs, 16)   # first layer
        self.relu = nn.ReLU()             # activation
        self.l2 = nn.Linear(16, 1)        # output layer

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x

model = MyNet(X_train.shape[1])

# set up loss function and optimizer

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#loop

epochs = 800
best_val_loss=float("inf")
for i in range(epochs):
    model.train()
    outputs = model(X_train)
    loss = loss_function(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # every 20 epochs, check validation data
    if (i + 1) % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = loss_function(val_out, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            print("Early stopping at epoch", i + 1)
            break

        pred = val_out.detach().numpy()
        actual = y_val.detach().numpy()

        percentage_error = abs((pred - actual) / (actual + 1e-8)) * 100

        mean_percentage_error = percentage_error.mean()

        print("Epoch:", i + 1,
              "Train loss:", float(loss),
              "Val loss:", float(val_loss),
              "Mean Error:", float(mean_percentage_error))

# predict

model.eval()
with torch.no_grad():
    sample = X_val[:5]
    pred = model(sample)

    print("Predicted PDSI values:", pred.numpy().flatten())