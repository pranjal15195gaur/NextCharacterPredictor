import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from latex import latexify, format_axes
from pprint import pprint
from sklearn.manifold import TSNE
import torch._dynamo
import warnings
warnings.filterwarnings("ignore")
torch._dynamo.config.suppress_errors = True

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Supported Device: {device}\n\n")


### MODEL ARCHITECTURE
class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.tanh(self.lin1(x))
    x = self.lin2(x)
    return x



### LOAD THE CORPUS
fileWonder = open("wonderland.txt", "r")
wonder = fileWonder.read()
print(wonder[:1000])
new_wonder = ""
for char in wonder:
    if char in ['ù', '—', '‘', '’', '“', '”']:
        continue
    new_wonder += char

characters = sorted(list(set(new_wonder)))
print(f"Total Characters in the Corpus: {len(new_wonder)}\n")
print(f"Number of Characters in the Corpus: {len(characters)}\n")
print(f"The characters are:\n{characters}\n\n")


### MAPPING OF CORPUS CHARACTER TO INDEX AND VICE VERSA
stoi = {s : i + 1 for i, s in enumerate(characters)}
stoi["+"] = 0 ## Pad Character
itos = {i : s for s, i in stoi.items()}
print(f"Dictionary Mapping of indices to characters:\n{itos}\n\n")


### CREATING TRAINING SAMPLES
block_size = 100
X, Y = [], []
context = [0] * block_size
for idx in range(len(new_wonder)):
  ix = stoi[new_wonder[idx]]
  X.append(context)
  Y.append(ix)
  # print(''.join(itos[i] for i in context), '--->', itos[ix])
  context = context[1:] + [ix]

X = torch.tensor(X).to(device)
Y = torch.tensor(Y).to(device)

print(f"Training Samples: {X.shape}\nLabels:{Y.shape}\n\n")


### EMBEDDING LAYER
emb_dim = 256
emb = torch.nn.Embedding(len(stoi), emb_dim)
print(f"Embeddings Shape: {emb.weight.shape}\n\n")


def plot_emb(emb, itos, ax = None):
    if emb.weight.shape[1] != 2:
      tsne = TSNE(n_components = 2)
      emb_new = tsne.fit_transform(emb.weight.detach().cpu().numpy())
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(len(itos)):
        if emb.weight.shape[1] == 2:
          x, y = emb.weight[i].detach().cpu().numpy()
          ax.scatter(x, y, color='k')
          ax.text(x + 0.05, y + 0.05, itos[i])
          ax.set_title("2D Embeddings Before Training")
        else:
          x, y = emb_new[i]
          ax.scatter(x, y, color = 'k')
          ax.text(x + 0.05, y + 0.05, itos[i])
          ax.set_title("2D Embeddings using t-SNE After Training")
    return ax



model = NextChar(block_size, len(stoi), emb_dim, 100).to(device)
model = torch.compile(model)
print("Model Architecture:\n")
for param_name, param in model.named_parameters():
    print(param_name, param.shape)


### INFERRING FROM THE MODEL
g = torch.Generator()
g.manual_seed(4200)
def generate_text(model, itos, stoi, block_size, max_len, start_str = None):

    context = [0] * block_size
    if start_str:
        for s in start_str:
            context = context[1:] + [stoi[s]]
    text = start_str if start_str else ""
    for i in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits = y_pred).sample().item()
        ch = itos[ix]
        text += ch
        context = context[1:] + [ix]
    return text



start = np.random.randint(0, len(new_wonder) - block_size - 1)
end = start + block_size
while new_wonder[start] != " ":
  start += 1

while new_wonder[end] != " ":
  end -= 1

seed_text = new_wonder[start + 1 : end]

my_str = generate_text(model, itos, stoi, block_size, 1000, seed_text)
old_text = bytes(my_str, "utf-8").decode("unicode_escape")
print("\n======================================================\n")
print("Generated Text from Untrained Model: \n")
print(old_text)

device = torch.device("cpu")
model.load_state_dict(torch.load("modelWonder.pth", map_location = device))


print(f"\n==================Seed Text=================\n{seed_text}\n")
my_str = generate_text(model, itos, stoi, block_size, 1000, seed_text)
decoded_string = bytes(my_str, "utf-8").decode("unicode_escape")
print(f"\n===============Predicted Text===============\n{decoded_string}")



latexify(columns=2, fig_width=10)

# Plotting the embeddings before training
ax1 = plt.subplot(1, 2, 1)
plot_emb(emb, itos, ax=ax1)
format_axes(ax1)

# Plotting the embeddings after training
ax2 = plt.subplot(1, 2, 2)
plot_emb(model.emb, itos, ax=ax2)
format_axes(ax2)

plt.tight_layout()
plt.show()