# Opens the Syntehtic QA spreadsheet
import pandas as pd

data = pd.read_excel('SyntheticQA.xlsx')
questions = data[data.columns[1]]
answers = data[data.columns[2]]

# Accesses OpenAI API
from openai import OpenAI
client = OpenAI(api_key="Open AI Key")
