import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("C:/Users/Subham/Desktop/Learn N Build/Machine learning algorithm by kris sir/Assignments to do/IRIS DATA SET/IRIS.csv")
print("this is the code for plotting a scatter  plot of iris data set of sepal length and width with their species")
colors={'Iris-setosa':'red','Iris-versicolor':'blue','Iris-virginica':'green'}
plt.scatter(data['sepal_length'],data['sepal_width'],c=data['species'].map(colors),label=[colors.keys()])
plt.legend()
plt.show

plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.title('All the species according to sepal length&width')
plt.show()

#data column greater than speal length mean
x=data['petal_length']
a=x.mean()
print(a)
data_greater_than_mean=x[x>a]
print(data_greater_than_mean)
