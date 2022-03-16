import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('data/curated_freq.csv')

ax = df.hist(column='vid_count', grid=False, bins=100, figsize=(8,10), layout=(3,1), sharex=True, color='#86bf91', zorder=2, rwidth=0.9)
plt.show()