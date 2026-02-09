import numpy as np

class Perceptron:
    def __init__(self, input_size=0, learning_rate=0.1):
        self.X = np.array([])
        self.w = np.zeros(input_size + 1) # threshold w[0]
        self.learning_rate = learning_rate
        self.target = np.array([])
        self.error = 0
    
    def set_X(self, X: np.array) -> None:
        X = np.array(X)
    
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim != 2:
            print("X must be 1D or 2D array")
            return
        
        self.X = X
        self.X = np.insert(self.X, 0, -1, axis=1)

    def set_w(self, w: np.array) -> None:
        self.w = w          # the first one is a threshold

    def set_target(self, target: np.array) -> None:
        if self.X.ndim != 2:
            print("X not setted!")
            return
        
        if len(target) != len(self.X):
            print(f"Invalid size of target vector. It must have length = {len(self.X)}. Now = {len(target)}")
            return
        
        self.target = target
    
    def activate(self, arr_wsum: np.array) -> np.array:
        return 2 / (1 + np.exp(-arr_wsum)) - 1
    
    def forward(self) -> np.array:
        if self.X.size == 0:
            print("Input X vector not set!")
            return None
        
        wsum = np.dot(self.X, self.w)
        y = self.activate(wsum)

        return y
    
    
    

        
        

        
        



        
    

