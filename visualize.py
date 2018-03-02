import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
style.use('ggplot')

df = pd.read_csv('./dataset/sum.csv', sep='\|\|', engine='python')
df = df.sort_values(by=['sentiment'])
df = df.loc[df['sentiment'].isnull() == False]

print('Total number of record:', len(df['sentiment']))
print('Negative:', len(df.loc[df['sentiment'] == 0]))
print('Positive:', len(df.loc[df['sentiment'] == 1]))

X = [df.loc[df['sentiment'] == 0]['sentiment'], df.loc[df['sentiment'] == 1]['sentiment']]
plt.hist(X, bins=2)
plt.show()