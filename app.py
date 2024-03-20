import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

if (torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
    
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



st.write("""
         # Next $k$ character predictor
         ### Choose amongst the following corpuses and generatte the next $k$ characters
         """)
st.sidebar.title("Next $k$ character predictor")
st.sidebar.caption("App created by team TensionFlow using Streamlit as a part of the Machine Learning Course ES335")

select = st.selectbox("Select the Corpus", ["Gulliver's Travels", "Atomic Habits: James Clear", "Tolstoy's War and Peace", "Alice in the Wonderland"])

seed_number = st.slider("Choose the Seed Number", 0, 10000)
k = st.slider("Number of Characters to be generated $k$", 50, 2000)

option = st.radio("Generate the Seed Text?", ("Yes", "No"))

g = torch.Generator()
g.manual_seed(seed_number)

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
        

def stream_data(str):
        for word in str.split(" "):
            yield word + " "
            time.sleep(0.03)
            
if (select == "Gulliver's Travels"):
    fileGulliver = open("gulliver.txt", "r")
    gulliver = fileGulliver.read()
    new_gulliver = ""
    for char in gulliver:
        if char in ['$', '%', '&', 'æ', 'œ', '–', '—', '‘', '’', '“', '”', '•', '™']:
            continue
        new_gulliver += char.lower()

    characters = sorted(list(set(new_gulliver)))
    stoi = {s : i + 1 for i, s in enumerate(characters)}
    stoi["+"] = 0 ## Pad Character
    itos = {i : s for s, i in stoi.items()}
    
    block_size = 20
    emb_dim = st.selectbox("Select the Embedding Dimension (Block Size = 20 Fixed)", [15, 25, 50])

    class NextChar(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size1, hidden_size2):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.lin1 = nn.Linear(block_size * emb_dim, hidden_size1)
            self.lin2 = nn.Linear(hidden_size1, hidden_size2)
            self.lin3 = nn.Linear(hidden_size2, vocab_size)

        def forward(self, x):
            x = self.emb(x)
            x = x.view(x.shape[0], -1)
            x = torch.tanh(self.lin1(x))
            x = torch.tanh(self.lin2(x))
            x = self.lin3(x)
            return x
        
    if (option == "No"):
        seed_text = st.text_input("Enter the seed text (no digits)")
    else:
        l = st.slider("Select the length of seed_text", 20, k)
        
        start = np.random.randint(0, len(new_gulliver) - block_size - 1)
        end = start + l
        while new_gulliver[start] != " ":
            start += 1

        while new_gulliver[end] != " ":
            end -= 1

        seed_text = new_gulliver[start + 1 : end]
    
    
    btn = st.button("Generate Text")
    if btn:
        st.subheader("Seed Text")
        st.write_stream(stream_data(seed_text))
        model = NextChar(block_size, len(stoi), emb_dim, 100, 75).to(device)
        model = torch.compile(model)
        if emb_dim == 15:
            model.load_state_dict(torch.load("modelGulliver20_15.pth", map_location = device))
        elif emb_dim == 25:
            model.load_state_dict(torch.load("modelGulliver20_25.pth", map_location = device))
        elif emb_dim == 50:
            model.load_state_dict(torch.load("modelGulliver20_50.pth", map_location = device))
        my_str = generate_text(model, itos, stoi, block_size, k, seed_text)
        decoded_string = bytes(my_str, "utf-8").decode("unicode_escape")
        st.header("Generated Text")
        st.write_stream(stream_data(decoded_string))
        st.sidebar.subheader("Seed Text")
        st.sidebar.write_stream(stream_data(seed_text))
        st.sidebar.header("Generated Text")
        st.sidebar.write_stream(stream_data(decoded_string))
        
elif (select == "Tolstoy's War and Peace"):
    
    fileTolstoy = open("tolstoy.txt", "r")
    tolstoy = fileTolstoy.read()

    new_tolstoy = ""
    for char in tolstoy:
        if char in ['à', 'ä', 'é', 'ê']:
            continue
        new_tolstoy += char.lower()

    characters = sorted(list(set(new_tolstoy)))
    
    stoi = {s : i + 1 for i, s in enumerate(characters)}
    stoi["_"] = 0
    itos = {i : s for s, i in stoi.items()}
    
    block_size = 100
    emb_dim = 256
    model = NextChar(block_size, len(stoi), emb_dim, 100).to(device)
    model = torch.compile(model)   
    if (option == "No"):
        seed_text = st.text_input("Enter the seed text (for alphanumeric characters, only lowercase allowed)")
    else:
        l = st.slider("Select the length of seed_text", 20, k)
        
        start = np.random.randint(0, len(new_tolstoy) - block_size - 1)
        end = start + l
        while new_tolstoy[start] != " ":
            start += 1

        while new_tolstoy[end] != " ":
            end -= 1

        seed_text = new_tolstoy[start + 1 : end]
        
    btn = st.button("Generate Text")
    if btn:
        st.subheader("Seed Text")
        st.write_stream(stream_data(seed_text))
        model.load_state_dict(torch.load("modelTolstoy.pth", map_location = device))
        my_str = generate_text(model, itos, stoi, block_size, k, seed_text)
        decoded_string = bytes(my_str, "utf-8").decode("unicode_escape")
        st.header("Generated Text")
        st.write_stream(stream_data(decoded_string))
        st.sidebar.subheader("Seed Text")
        st.sidebar.write_stream(stream_data(seed_text))
        st.sidebar.header("Generated Text")
        st.sidebar.write_stream(stream_data(decoded_string))
    
elif (select == "Alice in the Wonderland"):
    fileWonder = open("wonderland.txt", "r")
    wonder = fileWonder.read()
    new_wonder = ""
    for char in wonder:
        if char in ['ù', '—', '‘', '’', '“', '”']:
            continue
        new_wonder += char

    characters = sorted(list(set(new_wonder)))
    stoi = {s : i + 1 for i, s in enumerate(characters)}
    stoi["+"] = 0 ## Pad Character
    itos = {i : s for s, i in stoi.items()}
    
    block_size = 100
    emb_dim = 256
    model = NextChar(block_size, len(stoi), emb_dim, 100).to(device)
    model = torch.compile(model)
    if (option == "No"):
        seed_text = st.text_input("Enter the seed text (no digits)")
    else:
        l = st.slider("Select the length of seed_text", 20, k)
        
        start = np.random.randint(0, len(new_wonder) - block_size - 1)
        end = start + l
        while new_wonder[start] != " ":
            start += 1

        while new_wonder[end] != " ":
            end -= 1

        seed_text = new_wonder[start + 1 : end]
        
    btn = st.button("Generate Text")
    if btn:
        st.subheader("Seed Text")
        st.write_stream(stream_data(seed_text))
        model.load_state_dict(torch.load("modelWonder.pth", map_location = device))
        my_str = generate_text(model, itos, stoi, block_size, k, seed_text)
        decoded_string = bytes(my_str, "utf-8").decode("unicode_escape")
        st.header("Generated Text")
        st.write_stream(stream_data(decoded_string))
        st.sidebar.subheader("Seed Text")
        st.sidebar.write_stream(stream_data(seed_text))
        st.sidebar.header("Generated Text")
        st.sidebar.write_stream(stream_data(decoded_string))
        
elif (select == "Atomic Habits: James Clear"):
    fileAtomic = open("atomic.txt", "r")
    atomic = fileAtomic.read()
    new_atomic = ""
    for char in atomic:
        if char in ['\x0c', '£', 'Ü', 'ä', 'æ', 'è', 'é', 'ò', 'ó', 'ø', 'ü', '‐', '–', '—', '‘', '’', '“', '”', '−', '$', '%', '#', '&']:
            continue
        new_atomic += char.lower()

    characters = sorted(list(set(new_atomic)))

    stoi = {s : i + 1 for i, s in enumerate(characters)}
    stoi["~"] = 0 ## Pad character
    itos = {i : s for s, i in stoi.items()}
    
    block_size = 100
    emb_dim = 50
    emb = torch.nn.Embedding(len(stoi), emb_dim)
    
    
    class ImprovedNextChar(nn.Module):
        def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim)
            self.rnn = nn.LSTM(emb_dim, hidden_size, num_layers = 2, batch_first = True)
            self.fc = nn.Linear(hidden_size * block_size, vocab_size)

        def forward(self, x):
            x = self.emb(x)
            x, _ = self.rnn(x)
            x = x.contiguous().view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = ImprovedNextChar(block_size, len(stoi), emb_dim, 50).to(device)
    model = torch.compile(model)
    
    if (option == "No"):
        seed_text = st.text_input("Enter the seed text (for alphanumeric characters, only lowercase allowed)")
    else:
        l = st.slider("Select the length of seed_text", 20, k)
        
        start = np.random.randint(0, len(new_atomic) - block_size - 1)
        end = start + l
        while new_atomic[start] != " ":
            start += 1

        while new_atomic[end] != " ":
            end -= 1

        seed_text = new_atomic[start + 1 : end]
        
    btn = st.button("Generate Text")
    if btn:
        st.subheader("Seed Text")
        st.write_stream(stream_data(seed_text))
        model.load_state_dict(torch.load("modelAtomic.pth", map_location = device))
        my_str = generate_text(model, itos, stoi, block_size, k, seed_text)
        decoded_string = bytes(my_str, "utf-8").decode("unicode_escape")
        st.header("Generated Text")
        st.write_stream(stream_data(decoded_string))
        st.sidebar.subheader("Seed Text")
        st.sidebar.write_stream(stream_data(seed_text))
        st.sidebar.header("Generated Text")
        st.sidebar.write_stream(stream_data(decoded_string))
