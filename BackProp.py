import numpy as np
def sigmoid(x, derivative=False):

    if (derivative == True):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))



X = np.array([
    [0, 0, 0],  
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
   # [1, 1, 0],
    [1, 1, 1],
])

y = np.array([[0,0],
              [1,0],
              [1,0],
              [0,1],
              [1,0],
              [0,1],
         #     [0,1],
              [1,1]])


class network(object):
    def __init__(self,sizes):
        self.sizes = sizes
        self.numLayers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.a = [np.zeros(y) for y in sizes]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
    def feed_Forward(self,input_):
        if len(input_) != self.sizes[0]:
            raise ValueError('wrong number of inputs')
        
        self.a[0]= np.array(input_)
        for ind in range(1,len(self.sizes)):
            for i in range(self.sizes[ind]):
                    sum_ = np.dot(self.a[ind-1],self.weights[ind-1][i])
                    sum_+=self.biases[ind-1][i]
                    self.a[ind][i]=sigmoid(sum_)
        
        return(self.a[len(self.sizes)-1])
        
    def back_Propogate(self,target,N,M=0):
        #print(del)
        delta = ((self.a[len(self.sizes)-1] - target) * sigmoid(self.a[self.numLayers-1], derivative=True))
        self.biases[-1] = self.biases[-1]-N*np.matrix(delta).transpose()
        #print(delta)
        for i in range(len(delta)):
            self.weights[-1][i] =self.weights[-1][i]-N* np.dot(delta[i] ,self.a[-2])
        #self.weights[-1] = np.dot(delta,np.matrix(self.a[-2]))
        for l in range(2, self.numLayers):
            sp = sigmoid(self.a[-l],derivative=True)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            self.biases[-l] =self.biases[-l]-N* np.matrix(delta).transpose()
            #print("delta",delta)
            self.weights[-l] = self.weights[-l] -N* np.dot(delta, self.a[-l-1].transpose())
        
    def trainOne(self,input,target,learningRate=0.2,momentum=0):
        self.feed_Forward(input)
        self.back_Propogate(target,learningRate,momentum)
        error=self.calcError(target)
        return error

    def calcError(self,targets):
        error = 0.0
        for k in range(len(targets)):
            error = error + (targets[k]-self.a[-1][k])**2
        return np.sqrt(error/len(targets))

    def SGD(self,X,Y,epochs):
        error=[]
        if len(X) != len(Y):
            raise ValueError('X !=Y')
        for _ in range(epochs):
            er=[]
            for x_,y_ in zip(X,Y):
                er.append(self.trainOne(x_,y_))
            error.append(np.array(er).mean())
        return(error)
    
    def test_all(self,X):
        op=[]
        for x in X:
            op.append(np.round(self.feed_Forward(x)))
        return(op)



if __name__=="__main__":
    nn=network([3,3,2])
    err=nn.SGD(X,y,10000)
    # for i in range(1000):
    #     op=nn.feed_Forward([0,0,0])
    #     #network([4,5,5,1])
    #     nn.back_Propogate([0,1],0.2)
    op = nn.test_all(X)

