import pickle as pkl
with open('./genarated.pkl', 'rb') as f:
    x = pkl.load(f)
print(x)
