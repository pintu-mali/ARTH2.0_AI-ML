print("""
##########################
#                        #
#    Machine Learning    #
#                        #
##########################
""")

while True:
	print("""
	1.Linear Regression
	2.Logistic Regression
	3.Load a model
	""")
	ch = int(input("Enter your choice: "))
	if ch == 1:
		import numpy as np
		import pandas as pd
		import joblib
		import os
		os.system("tput setaf 21")
		print("""
		Our model will only work if your
		dataset has undergone preprocessing
		and 
		the output column name should be 'y'
		""")
		os.system("tput setaf 7")
		location = input("Enter the location of dataset: ")
		data = pd.read_csv(location)
		x = data.drop('y',axis=1)
		y = data[['y']]
		from sklearn.model_selection import train_test_split
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
		from sklearn.linear_model import LinearRegression
		model = LinearRegression()
		model.fit(x,y)
		model.predict(x)
		while True:
			ch = int(input("""
			1. To predict
			2. Bias 
			3. Weight 
			4. To exit
			Enter your choice: """))
			if ch == 1:
				x_input = (input("Enter your x values in order: "))
				y_pred = model.predict([list(x_input.split(","))])
				os.system("tput setaf 34")
				print(y_pred)
				os.system("tput setaf 7")
				
			elif ch == 2:
				bias = model.intercept_
				os.system("tput setaf 34")
				print(bias)
				os.system("tput setaf 7")

			elif ch == 3:
				os.system("tput setaf 34")
				weight = model.coef_
				print(weight)
				os.system("tput setaf 7")

			elif ch == 4:
				break

		save = input("want to save the model(y/n): ")
		if save == "y":
			os.system("tput setaf 1")
			print("Note:Extension should be 'pk1'")
			os.system("tput setaf 7")
			dump = input("Enter your model name: ")
			joblib.dump(model,"{}".format(dump))

	elif ch == 2:
		import numpy as np
		import pandas as pd
		import joblib
		import os
		os.system("tput setaf 21")
		print("""
		Our model will only work if your
		dataset has undergone preprocessing
		and 
		the output column name should be 'y'
		""")
		os.system("tput setaf 7")
		location = input("Enter the location of dataset: ")
		data = pd.read_csv(location)
		x = data.drop('y',axis=1)
		y = data[['y']]
		from sklearn.model_selection import train_test_split
		x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
		from sklearn.linear_model import LogisticRegression
		model = LogisticRegression()
		model.fit(x,y)
		model.predict(x)
		while True:
			ch = int(input("""
			1. To predict
			2. Bias 
			3. Weight 
			4. To exit
			Enter your choice: """))
			if ch == 1:
				x_input = (input("Enter your x values in order: "))
				y_pred = model.predict([list(x_input.split(","))])
				os.system("tput setaf 34")
				print(y_pred)
				os.system("tput setaf 7")
				
			elif ch == 2:
				bias = model.intercept_
				os.system("tput setaf 34")
				print(bias)
				os.system("tput setaf 7")

			elif ch == 3:
				os.system("tput setaf 34")
				weight = model.coef_
				print(weight)
				os.system("tput setaf 7")

			elif ch == 4:
				break

		save = input("want to save the model(y/n): ")
		if save == "y":
			os.system("tput setaf 1")
			print("Note:Extension should be 'pk1'")
			os.system("tput setaf 7")
			dump = input("Enter your model name: ")
			joblib.dump(model,"{}".format(dump))

	elif ch == 3:
		import numpy as np
		import pandas as pd
		import joblib
		import os
		location = input("Enter the location of pk1 file: ")
		model = joblib.load(location)
		y_pred = input("Enter x values in order: ")
		o = model.predict([list(y_pred.split(","))])
		print(o)
	elif ch == 4:
		break
