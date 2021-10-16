from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, filename, plotFileName):
    
    df = pd.DataFrame(data)
    print(df)

    X , y = prepare_data(df)
    print(f"X shape :{X.shape}, y shape : {y.shape}")
    
    model_and = Perceptron(eta = eta, epochs=epochs)
    model_and.fit(X,y)

    _ = model_and.total_loss()

    save_model(model_and, filename=filename)
    save_plot(df,file_name=plotFileName, model=model_and)


if __name__ == '__main__':
    AND ={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y" :[0,0,0,1]
    }
    ETA = 0.01
    EPOCH = 10
    main(data=AND, eta=ETA, epochs=EPOCH,filename= "and.model",plotFileName= "and.png")