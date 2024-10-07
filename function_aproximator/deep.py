
import torch

"""
Modelo de RN para espacios vectoriales de muchas dimensiones (ej imágenes, miles de pixeles)
"""

class DeepActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        CNN profunda que producirá 2 valores continuos (media y desviación tìpica)
        input_shape -> observaciones del actor
        output_shape -> acciones que debe producir el actor
        """
        super(DeepActor, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(512, output_shape)
        self.actor_sigma = torch.nn.Linear(512, output_shape)
        
    def forward(self, x):
        #Return media y desviación estandar para política gaussiana
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma
        
class DeepDiscreteActor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        CNN que utilizará una función logística para discriminar la acción del espacio de acciones discreto
        input_shape -> observaciones del actor
        output_shape -> acciones que debe producir el actor
        """
        
        super(DeepDiscreteActor, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.actor_logits = torch.nn.Linear(512, output_shape)
        
    def forward(self, x):
        #Return calculamos la acción a tomar
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        logits = self.actor_logits(x)
        return logits
        
class DeepCritic(torch.nn.Module):
    def __init__(self, input_shape, output_shape = 1, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        CNN que produce un valor continuo, estima el valor de la observación / estado actual
        input_shape -> observaciones del critic
        output_shape -> acciones que debe producir del critic
        """
        
        super(DeepCritic, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.critic = torch.nn.Linear(512, output_shape)
        
    def forward(self, x):
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        critic = self.critic(x)
        return critic
        

        
class DeepActorCritic(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        CNN que se utiliza para representar tanto al actor como al crítico en el algoritmo A2C
        input_shape -> observaciones del actor
        actor_shape -> forma de los datos de salida del actor
        critic_shape -> forma de los datos de salida del critic
        """
        
        super(DeepActorCritic, self).__init__()
        self.device = device
        
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 32, 8, stride = 4, padding = 0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride = 2, padding = 0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, stride = 1, padding = 0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(64 * 7 * 7, 512),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(512, actor_shape)
        self.actor_sigma = torch.nn.Linear(512, actor_shape)
        self.critic = torch.nn.Linear(512, critic_shape)
        
    def forward(self, x):
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic