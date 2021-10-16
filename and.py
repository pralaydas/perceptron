from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np


AND ={
    "x1":[0,0,1,1],
    "x2":[0,1,0,1],
    "y" :[0,0,0,1]
}
df = pd.DataFrame(AND)
print(df)

X , y = prepare_data(df)
print(f"X shape :{X.shape}, y shape : {y.shape}")
ETA = 0.01
EPOCH = 10
model = Perceptron(eta = ETA, epochs=EPOCH)
model.fit(X,y)

_ = model.total_loss()

save_model(model, filename="and.model")
save_plot(df,file_name="and.png", model=model)