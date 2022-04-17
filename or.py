from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(data)
    print(df)

    X , y = prepare_data(df)
    print(f"X shape :{X.shape}, y shape : {y.shape}")
    
    model_or = Perceptron(eta = eta, epochs=epochs)
    model_or.fit(X,y)

    _ = model_or.total_loss()

    save_model(model_or, filename=filename)
    save_plot(df,file_name=plotFileName, model=model_or)

if __name__ == '__main__':
    AND ={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y" :[0,1,1,1]
    }
    ETA = 0.01
    EPOCH = 10
    main(data=AND, eta=ETA, epochs=EPOCH,filename= "or.model",plotFileName= "or.png")