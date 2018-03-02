import pandas as pd

df = pd.read_csv('./dataset/sum.csv', sep='\|\|', engine='python')
df = df.sort_values(by=['sentiment'])
df = df.loc[df['sentiment'].isnull() == False]

def getTurningPoint():
	prev = 0
	count = 0
	turn = 0
	for d in df['sentiment']:
		if d != prev:
			turn = count

		prev = d
		count += 1

	return turn
