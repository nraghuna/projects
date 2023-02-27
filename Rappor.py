import numpy as np
class Rappor():
    def __init__(self, p, q):
        self.p = p
        self.q = q

    def encode(self,v,d):
        B= np.zeros(d)
        for i in range(len(B)):
            if i !=v:
                B[i]==0
            else:
                B[i]=1
        return B

    def perturb(self,f,B,d):
        self.prob=[]
        B1= np.zeros(d)
        for i in range(len(B)):
            if B[i]==1:
                B1[i]= 1-0.5*f
            else:
                B1[i]=1.5*f
            self.prob.append(B1[i])

    def instantaneous(self,f,B,d):
        B1 = np.zeros(d)
        for i in range(len(B)):
            if B[i] == 1:
                s = self.p
            else:
                s = self.q
            res = random.random()
            if (res < s):
                B1[i] = 1
            else:
                B1[i] = 0
        return B1



