import pickle as pkl
with open('generated.pkl', 'rb') as f:
    x = pkl.load(f)
print(x)
