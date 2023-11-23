from bs4 import BeautifulSoup
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
from tkinter import *
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import spacy
from scipy import stats

# Scraping Data
url = 'https://wikimon.net/Visual_List_of_Digimon'
x = requests.get(url)
soup = BeautifulSoup(x.content, 'html.parser')
names = []
images = []

for a in soup.find_all('table', style="text-align: center; width: 130px; float: left; margin: 0px 4px 2px 0px; background-color: #222222;"):
    for b in a.find_all('a'):
        nama = b.text
        if nama != '':
            names.append(nama)
        
    for c in a.find_all('img'):
        judul = c.get('src')
        if judul is not None:
            images.append('https://wikimon.net/' + judul)

# Creating a DataFrame using Pandas
df = pd.DataFrame({'nama': names, 'gambar': images})

# Save as CSV
df.to_csv('5.digimondict.csv', index=False, encoding='utf-8')

# Display Data
print(df.head())

# Data Visualization using Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(y='nama', data=df, order=df['nama'].value_counts().index)
plt.title('Distribution of Digimon Names')
plt.xlabel('Count')
plt.ylabel('Digimon Names')
plt.show()

# Import to MySQL
mydb = mysql.connector.connect(
    host='localhost',
    user='username',
    passwd='password',
    database='digimon'
)
cursor = mydb.cursor()

# Clear existing data
cursor.execute('DELETE FROM digimon')
cursor.execute('ALTER TABLE digimon AUTO_INCREMENT = 1')

# Insert new data
for index, row in df.iterrows():
    cursor.execute('INSERT INTO digimon (nama, gambar) VALUES (%s, %s)', (row['nama'], row['gambar']))

mydb.commit()
cursor.close()

# PyTorch Example: Simple Linear Regression
# Generating some example data
torch.manual_seed(42)
X = torch.rand(100, 1) * 10
y = 2 * X + 1 + torch.randn(100, 1)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Train the model
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Test the model with new data
X_test = torch.tensor([[1.0], [2.0], [3.0]])
y_pred_test = model(X_test)
print("Predictions:", y_pred_test.detach().numpy())

# Scipy Example: Correlation Test
# Example data
x = df['example_column_x']
y = df['example_column_y']
correlation, p_value = stats.pearsonr(x, y)
print("Correlation:", correlation)
print("P-value:", p_value)

# Scikit-learn Example: Train-Test Split
# Example data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Spacy Example: Tokenization
nlp = spacy.load("en_core_web_sm")
text = "This is an example sentence."
doc = nlp(text)
print("Tokens:", [token.text for token in doc])

# Tkinter Example: Display a Message Box
root = Tk()
root.withdraw()  # Hide the main window
messagebox.showinfo("Information", "Data processing completed!")

# Tkinter Example: Simple GUI
def click_button():
    messagebox.showinfo("Button Clicked", "Hello, Tkinter!")

root = Tk()
root.title("Digimon Data Processing")

button = Button(root, text="Click Me", command=click_button)
button.pack(pady=20)

root.mainloop()
