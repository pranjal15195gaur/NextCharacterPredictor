# Next Character Predictor
## Find the app hosted at [Link](https://tensionflowcharpredict.streamlit.app/)
**Datasets: [Link](https://cs.stanford.edu/people/karpathy/char-rnn/)**

### **Embedding the characters as a Vector $\in \mathbb{R}^N$**
![](https://github.com/guntas-13/NextCharacterPredictor/blob/main/Embed.svg)

### **Input layer of the model and creating the training examples**
![](https://github.com/guntas-13/NextCharacterPredictor/blob/main/ModelEmbed.svg)
![](https://github.com/guntas-13/NextCharacterPredictor/blob/main/MLPToken.svg)

To make inference from the model trained on ```tolstoy.txt``` run the script:
```bash
python3 predictT.py
```

To make inference from the model trained on ```wonderland.txt``` run the script:
```bash
python3 predictW.py
```

To run the StreamLit App, run the script:
```bash
streamlit run app.py
```
