import numpy as np

def create_sequences(df, seq_length, bert_emb):
    xemb, x_pr, ys = [], [], []
    # Iterate over data indices
    for i in range(len(df) - seq_length):
      	# Define inputs
        xemb.append(bert_emb[i:i+seq_length])
        x_pr.append(df.iloc[i:i+seq_length, 0].values.reshape(-1, 1))
        
        # x = np.concatenate((bert_emb[i:i+seq_length], df.iloc[i:i+seq_length, 0].values.reshape(-1, 1)), axis=1)
        # Define target
        y = df.iloc[i+seq_length, 0]
        # xs.append(x)
        ys.append(y)
    return np.array(xemb), np.array(x_pr).squeeze(), np.array(ys)

def create_sequences_sent(df, seq_length, bert_emb, fingpt_sentiments):
    xemb, x_pr, x_sent, ys = [], [], [], []
    # Iterate over data indices
    for i in range(len(df) - seq_length):
      	# Define inputs
        xemb.append(bert_emb[i:i+seq_length])
        x_pr.append(df.iloc[i:i+seq_length, 0].values.reshape(-1, 1))
        x_sent.append(fingpt_sentiments[i:i+seq_length])
        
        # xemb = np.concatenate((xemb, fingpt_sentiments[i:i+seq_length].reshape(-1, 1)), axis=1)
        # Define target
        y = df.iloc[i+seq_length, 0]
        # xs.append(x)
        ys.append(y)
    return np.array(xemb), np.array(x_pr).squeeze(), np.array(x_sent), np.array(ys)