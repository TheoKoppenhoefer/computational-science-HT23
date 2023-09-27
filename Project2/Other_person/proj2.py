import numpy as np 
import matplotlib.pyplot as plt 
import numpy.linalg as npl 
class Mouse:
    def __init__(self):
        """
        Used to generating parameters and initial vector
        """
        self.pn = 1 #p_n  
        self.po = 1 #p_o 
        self.pt = 1 #p_t
        self.lif = 0.06 #lif 
        self.n = 2 #dim 
        self.kd = 0.1 #param 
        self.ko = 0.3 #param 
        self.knt = 0.2 #param 
        self.To = 0.05 #overexpression 
        self.NOo = 0.3 #overexpression 

        self.x0 = self.x0() #initial values 
    
    def NT(self,N,T):
        """
        Used to generate the [N|T] expression 
        """
        term1 = 1/2*(self.kd + N + T) 
        term2 = np.sqrt(term1**2 - N*T) 
        return term1 - term2 
    
    def ANT(self,NT):
        """
        Used to generate A([N|T],X_total)
        """
        return (NT/self.knt)**self.n / (1 + (NT/self.knt)**self.n) 

    def part(self, vec):
        """
        Partial derivatives without expression moments 
        """
        N,O,T = vec

        NT = self.NT(N,T)
        ANT = self.ANT(NT)

        fraco = (O/self.ko)/(1 + O/self.ko ) 

        v1 = self.pn*fraco - N 
        v2 = self.po*fraco*ANT - O 
        v3 = self.pt*fraco*ANT - T 

        return np.array([v1,v2,v3]) 
    
    def x0(self):
        """
        Initial value
        """
        return np.zeros(3)
    
    def MEF(self):
        """
        MEF vector 
        """
        return np.array([0,0,self.To]) + self.lif*np.zeros(3)
    
    def NO(self):
        """
        NANOG Overexpression vector 
        """
        return np.array([self.NOo,0,self.To]) + self.lif*np.array([1,1,0])
    
    def OO(self):
        """
        OCT4 Overexpression vector 
        """
        return np.array([0,self.NOo,self.To]) + self.lif*np.array([1,1,0])
    
    def TO(self):
        """
        Tet1 Overexpression vector 
        """
        return np.array([0,0,self.NOo]) + self.lif*np.array([1,1,0]) 
    
    def ee(self,vec,f,dt):
        """
        Explicit Euler formula 
        """
        return vec + dt*f(vec)

    def RK4(self,vec,f,dt):
         """
         RK4 formula 
         """
         k1 = dt*f(vec)
         k2 = dt*f(vec + k1/2)
         k3 = dt*f(vec + k2/2)
         k4 = dt*f(vec + k3)
         upd = vec + 1/6*(k1 + 2*(k2 + k3) + k4)
         return upd 
    
    def evolve(self,t0=0.,tf=1000,dt=0.01):
        """
        Time evolution 
        """
        t = t0  #initial time 
        tl = [t0] #storage of time 
        vl = [self.x0] #storage of vector 
        vec = self.x0[:] #first vector 
        t+=dt #new time-step 
        mode = [self.MEF(), self.NO(), self.OO(), self.TO()] #expression modes 
        dv = lambda vec,id: self.part(vec) + mode[id] #generation of the partial derivatives 
        key = 0 #keys used for modes 
        seq = [0,2,0,0,1,0,0,3,0] #sequence of modes 
        kv = 1 #interval of change (a mode is sequenced into 3 steps)
        while t<tf and key<len(seq): #evolution 
            mod = seq[key] #mode 
            dvr = lambda v: dv(vec=v,id=mod) #mode-specific derivative 
            diff = dvr(vec) #func-eval of the derivative 
            vec = self.RK4(vec, dvr, dt) #update step 
            vl.append(vec) #appending step  
            tl.append(t) #appending time 
            t+=dt #new time 
            if npl.norm(diff,2)<1e-8: #stable-state for a mode 
                key+=1 #unlock new mode 
                if key//(3*kv)==1: #check if a new mode is sequenced 
                    vec=self.x0[:] #reset the vector 
                    kv+=1 #new condition for sequenced key 
                else: 
                    pass 
            else:
                pass 


        return vl, tl #returns expression vector and time vector 
    
    def plotting(self):
        labels = ["NANOG", "OCT4", "TET1"] #labels for plotting 
        v,t = self.evolve()
        plt.plot(t,v) #plotting 
        plt.grid()
        plt.legend(title="legend:", labels=labels)
        plt.show()
    



if __name__ == "__main__":
    Mouse().plotting() 