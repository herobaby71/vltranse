from utils.reldb import RelDB
import matplotlib.pyplot as plt

dataset = RelDB()
for i in range(10):
    item = dataset[i]
