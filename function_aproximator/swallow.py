

import torch

"""
Modelo de RN para espacios vectoriales de pocas dimensiones
"""

class SwallowActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Red neuronal q producirá 2 valores continuos (media y desviación tìpica) para cada uno de los valores output_shape
        input_shape -> observaciones del actor
        output_shape -> acciones que debe producir el actor
        """
        super(SwallowActor, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        
        self.actor_mu = torch.nn.Linear(32, output_shape)
        self.actor_sigma = torch.nn.Linear(32, output_shape)
        
    def forward(self, x):
        #Return media y desviación estandar para política gaussiana
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma
        
class SwallowDiscreteActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        RN que utilizará una función logística para discriminar la acción del espacio de acciones discreto
        input_shape -> observaciones del actor
        output_shape -> acciones que debe producir el actor
        """
        
        super(SwallowDiscreteActor, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        self.actor_logits = torch.nn.Linear(32, output_shape)
        
    def forward(self, x):
        #Return calculamos la acción a tomar
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        logits = self.actor_logits(x)
        return logits
        
class SwallowCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape = 1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        RN que produce un valor continuo, estima el valor de la observación / estado actual
        input_shape -> observaciones del critic
        output_shape -> acciones que debe producir del critic
        """
        
        super(SwallowCritic, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        self.critic = torch.nn.Linear(32, output_shape)
        
    def forward(self, x):
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        critic = self.critic(x)
        return critic
        

        
class SwallowActorCritic(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        RN que se utiliza para representar tanto al actor como al crítico en el algoritmo A2C
        input_shape -> observaciones del actor
        actor_shape -> forma de los datos de salida del actor
        critic_shape -> forma de los datos de salida del critic
        """
        
        super(SwallowActorCritic, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 32),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(32, 16),
                                          torch.nn.ReLU())
        
        self.actor_mu = torch.nn.Linear(16, actor_shape)
        self.actur_sigma = torch.nn.Linear(16, actor_shape)
        self.critic = torch.nn.Linear(16, critic_shape)
        
    def forward(self, x):
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic