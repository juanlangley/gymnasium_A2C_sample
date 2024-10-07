
import torch

def calculate_n_step_return(self, n_step_rewards, final_state, done, gamma):
    """
    Calcula el valor de retorno dados n-pasos para cada uno de los estados de entrada
    n_step_rewards -> Lista de las recompensas obtenidas en cada uno de los n estados
    final_state -> estado final tras las n iteraciones
    done -> variable booleana, true si se alcanzó el estado final del entorno
    gamma -> factor de descuento para el cálculo de la diferencia temporal
    
    return -> valor final de cada estado de los n ejecutados
    """
    
    g_t_n_s = list()
    with torch.no_grad()():
        g_t_n = torch.tensor([[0]]).float() if done else self.critic(self.preproc_obs(final_state)).cpu()
        for r_t in n_step_rewards[::-1]:
            g_t_n = torch.tensor(r_t).float() + gamma * g_t_n
            g_t_n_s.insert(0, g_t_n)
            
        return g_t_n_s