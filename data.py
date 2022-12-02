import csv 
from sklearn.model_selection import train_test_split

def generate_data(test_split=0.2, mode="both"):
	print("Generating new data...")
	f_train = open("data/train.txt", "w", encoding='utf-8')
	if mode == "both":
		f_valid = open("data/valid.txt", "w", encoding='utf-8')

	input_set = []
	label_set = []

	with open('data/train.csv', newline='', encoding='utf-8') as f: 
		reader = csv.reader(f)
		for row in reader:
			if row[-1] == "1":
				input_set.append(row[-3])
				label_set.append(row[-2])
	f.close()

	x_train, x_valid, y_train, y_valid = train_test_split(input_set, label_set, test_size=test_split, random_state=42)

	for x, y in zip(x_train, y_train):
		f_train.write(x + "\t\t" + y + "\n")

	if mode == "both":
		for x, y in zip(x_valid, y_valid):
			f_valid.write(x + "\t\t" + y + "\n")

	f_train.close()

	if mode == "both":
		f_valid.close()
	
	print("Finished.")

if __name__ == "__main__":
	generate_data()