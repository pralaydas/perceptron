from utils.model import Perceptron
import numpy as np
import pandas as pd
import joblib # to save my model as a binary file

inputs = np.array([[1,1]])


loaded_model = joblib.load("models/and.model")

ans =loaded_model.predict(inputs)
print(ans)