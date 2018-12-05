import pandas as pd
import numpy as np

dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C', 'D'])

print (df)
"""
                   A         B         C         D
2000-01-01  0.469112 -0.282863 -1.509059 -1.135632
2000-01-02  1.212112 -0.173215  0.119209 -1.044236
2000-01-03 -0.861849 -2.104569 -0.494929  1.071804
2000-01-04  0.721555 -0.706771 -1.039575  0.271860
2000-01-05 -0.424972  0.567020  0.276232 -1.087401
2000-01-06 -0.673690  0.113648 -1.478427  0.524988
2000-01-07  0.404705  0.577046 -1.715002 -1.039268
2000-01-08 -0.370647 -1.157892 -1.344312  0.844885

"""

#s = df['A']
#print(s[dates[5]])
print("======access column by index")
print(df.ix[[1]])
print("======access row by index")
print(df.iloc[0])
print(df.ix[[1]])
print("======access cell by index")
print(df.iloc[0][0])

print("======to array")
iv=df.index.tolist()
dfv = df.values
print(iv[0])
print(dfv[0][0], dfv.dtypes)


