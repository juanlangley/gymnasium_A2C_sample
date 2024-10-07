
from collections import namedtuple
import random
from collections import deque

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])


class ExperienceDeque(object):
    def __init__(self, capacity = int(1e6)):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def store(self, exp):
        self.memory.append(exp)
        
    def get_size(self):
        return len(self.memory)
    
    def continuous_batch(self, batch_size):
          batch = []
          if batch_size > len(self.memory):
              batch_size = len(self.memory)
              
          # como el deque siempre almacena las ultimas observaciones al final, para llenar el batch
          # recorremos esas observaciones y las agregamos al batch
          for i in range(len(self.memory) - batch_size, len(self.memory)):
              batch.append(self.memory[i]) 
          return batch
      
    def random_batch(self, batch_size):
        assert batch_size <= self.get_size(), "El tamaño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory, batch_size)
    

class ExperienceMemory(object):
    def __init__(self, capacity = int(1e6)):
        self.capacity = capacity
        self.memory_idx = 0
        self.memory = []
        
    def sample(self, batch_size):
        assert batch_size <= self.get_size(), "El tamaño de la muestra es superior a la memoria disponible"
        return random.sample(self.memory, batch_size)
    
    def get_size(self):
        return len(self.memory)
        
    
    def store(self, exp):
        self.memory.insert(self.memory_idx % self.capacity, exp)
        self.memory_idx += 1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        