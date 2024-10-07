

import gymnasium as gym
import numpy as np
import os
import random
import gymnasium as gym

from collections import namedtuple
from datetime import datetime
from argparse import ArgumentParser

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
import torch.nn.functional as F


from function_aproximator.swallow import SwallowActor
from function_aproximator.swallow import SwallowDiscreteActor
from function_aproximator.swallow import SwallowCritic
from function_aproximator.deep import DeepActor
from function_aproximator.deep import DeepDiscreteActor
from function_aproximator.deep import DeepCritic

from utils.params_manager import ParamsManager
from envs import atari
from envs import utils
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


args = ArgumentParser("DeepA2CAgent")
args.add_argument("--params_file", help = "Path del fichero JSON de parámetros", 
                  default="parameters.json", metavar="PFILE")
args.add_argument("--env", help= "Entorno de ID de Atari en Gym", default = "SeaquestNoFreameskip-v4", metavar = "ENV")
args.add_argument("--gpu_id", help= "ID de la GPU a utilizar", default = 0,
                  type=int, metavar="GPU_ID")
args.add_argument("--test", help="Modo de testing", action="store_true", default=False)
args.add_argument("--render", help="Renderiza el entorno en pantalla", action="store_true", default=False)
args.add_argument("--record", help="Almacena videos y estados de la performance del agente", action="store_true", default=False)
args.add_argument("--output_dir", help= "Directorio para almacenar outputs", default = "./trained_models/results")

args = args.parse_args()



manager = ParamsManager(args.params_file)
#ficheros de logs de configuración
summary_filename_prefix = manager.get_agent_params()["summary_filename_prefix"]
summary_filename = summary_filename_prefix + args.env + datetime.now().strftime("%y-%m-%d-%H-%M")

writer = SummaryWriter(summary_filename)

manager.export_agent_params(summary_filename + "/" + "agent_params.json")
manager.export_environment_params(summary_filename + "/" + "environment_params.json")





#habilitar gpu
use_cuda = manager.get_agent_params()["use_cuda"]
device = torch.device("cuda:"+ str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

#semilla para reproducir experimento
seed = manager.get_agent_params()["seed"]
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)
    

# Trayectoria ) [T1, T2, T3, T4, T5, T6]
# T_t = (st, at, rt, st+1)
Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])    

class DeepA2CAgent(mp.Process):
    def __init__(self, ID, env_name, agent_params, env_params):
        """
        ID -> identificador entero para identificar al agente en caso de tener múltiples instancias de agentes
        env_name -> nombre del entorno de aprendizaje
        agent_params -> parametros que usará el agente
        env_params -> parametros del entorno
        """
        
        super(DeepA2CAgent,self).__init__()
        
        self.ID = ID
        self.actor_name = "Actor "+str(self.ID)
        self.env_name = env_name
        self.env_params = env_params
        self.params = agent_params
        
        self.policy = self.multi_variate_gaussian_policy
        self.gamma = self.params['gamma']
        
        
        self.trajectory = [] #trayectoria del agente como secuencia de transiciones
        self.rewards = [] #recompensas obtenidas del entorno en cada paso
        
        self.global_step_num = 0
        
        self.best_mean_reward = -float("inf")
        self.best_reward = -float("inf")
        
        self.saved_params = False #si tengo parámetros guardados o no junto con el modelo
        self.continuous_action_space = True #para saber si el espacio de acciones es continuo o discreto
        
        
        

        
    def multi_variate_gaussian_policy(self, obs):
        """
        Calcula una distribución gaussiana multivariada del tamao de acciones usando las observaciones
        obs: observacion del agente
        return: policy, distribución sobre las acciones dadas las observaciones actuales
        """
        mu, sigma = self.actor(obs)
        value = self.critic(obs)
        # clamp de cada dimensión de mu basandonos en los limites de los espacios vectoriales de acciones (low, high)
        # x.clamp entre a,b mantiene a x entre los valores a y b
        [mu[:,i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i])) for i in range(self.action_shape)]
        #suavizar el valor de sigma
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7
        
        self.mu = mu.to(device)
        self.sigma = sigma.to(device)
        self.value = value.to(device)
        
        if len(self.mu.shape) == 0: #media es un escalar
            self.mu.unsqueeze_(0) # evita q la multiffariante normal de error
        self.action_distribution = MultivariateNormal(self.mu, torch.eye(self.action_shape)*self.sigma, validate_args=True)    
        return self.action_distribution
        
    def discrete_policy(self, obs):
        """
        Distribución discreta o categórica sobre las acciones dadas las observacinoes del agente
        obs: observaciones del agente
        return: Politica formada por una distribución sobre las acciones a partir de las observaciones
        """
        logits = self.actor(obs)
        value = self.critic(obs)
        self.logits = logits.to(device)
        self.value = value.to(device)
        self.action_distribution = Categorical(logits= self.logits)
        return self.action_distribution
        
    def process_action(self, action):
        if self.continuous_action_space:
            [action[:,i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i])) for i in range(self.action_shape)]
        action = action.to(device)
        return action.numpy().squeeze(0)
        
    def preprocess_obs(self, obs):
        obs = np.array(obs)
        if len(obs.shape) == 3:
            obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
            obs = np.resize(obs, (obs.shape[0], 84, 84))
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        return obs
    
    
    def get_action(self, obs):
        obs = self.preprocess_obs(obs)
        action_distribution = self.policy(obs)
        #print("ACTION_DIST -- ", action_distribution.logits )
        value = self.value
        #print("VALUE -- ", value[0] )
        action = action_distribution.sample()
        #print("ACTION -- ", action[0] )
        log_prob_a = action_distribution.log_prob(action)
        #print("PROB A -- ", log_prob_a[0] )
        action = self.process_action(action)
        #print("PROCESS ACTION-- ", action )
        
        if not self.params["test"]:
            self.trajectory.append(Transition(obs, value, action, log_prob_a))
        return action
        
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
        with torch.no_grad():
            g_t_n = torch.tensor([[0]]).float() if done else self.critic(self.preprocess_obs(final_state)).cpu()
            for r_t in n_step_rewards[::-1]:
                g_t_n = torch.tensor(r_t).float() + gamma * g_t_n
                g_t_n_s.insert(0, g_t_n)
                
            return g_t_n_s
        
    def calculate_loss(self, trajectory, td_targets):
        """
        Calculo de la perdida del crítico y del actor utilizando los td_targets y la trayectoria por otro
        
        """
        n_step_trajectory = Transition(*zip(*trajectory))
        v_s = n_step_trajectory.value_s
        log_prob_a = n_step_trajectory.log_prob_a
        actor_losses = []
        critic_losses = []
        
        for td_target, critic_prediction, log_p_a in zip(td_targets, v_s, log_prob_a):
            td_error = td_target - critic_prediction
            actor_losses.append(- log_p_a * td_error) # td_error es un estimador insesgado de Adventage
            critic_losses.append(F.smooth_l1_loss(critic_prediction, td_target))
            #critic_losses.append(F.mse_loss(critic_prediction, td_target))
        
        if self.params["use_entropy_bonus"]:
            actor_loss = torch.stack(actor_losses).mean() - self.action_distribution.entropy().mean()
        else:
            actor_loss = torch.stack(actor_losses).mean()
            
        critic_loss = torch.stack(critic_losses).mean()
        
        writer.add_scalar(self.actor_name + "/critic_loss", critic_loss, self.global_step_num)
        writer.add_scalar(self.actor_name + "/actor_loss", actor_loss, self.global_step_num)
        
        return actor_loss, critic_loss
        
    def learn(self, n_th_observation, done):
        td_targets = self.calculate_n_step_return(self.rewards, n_th_observation, done, self.gamma)
        actor_loss, critic_loss = self.calculate_loss(self.trajectory, td_targets)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph = True)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.trajectory.clear()
        self.rewards.clear()
        
    def save(self):
        file_name = self.params['model_dir']+"A2C_"+self.env_name+".ptm"
        agent_state = {"Actor": self.actor.state_dict(),
                       "Critic": self.critic.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, file_name)
        print("Estado del agente guardado en : ", file_name)
        
        if not self.saved_params:
            manager.export_agent_params(file_name + ".agent_params")
            print("Los parametros del agente se han guardado en ", file_name + ".agent_params")
            self.saved_params = True
        
        
    def load(self):
        file_name = self.params['model_dir']+"A2C_"+self.env_name+".ptm"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.actor.load_state_dict(agent_state["Actor"])
        self.critic.load_state_dict(agent_state["Critic"])
        self.actor.to(device)
        self.critic.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Cargado del modelo Actor-Critic desde", file_name,
              "que hasta el momento tiene una mejor recompensa media de: ",self.best_mean_reward,
              " y una recompensa máxima de: ", self.best_reward)
        
    def run(self):
        
        ## Cargar datos del entorno donde entrenar
        custom_region_available = False
        
        for key, value in self.env_params["useful_region"].items():
            if key in args.env:
                self.env_params["useful_region"] = value
                custom_region_available = True
                break
        
        if custom_region_available is not True:
            self.env_params["useful_region"] = self.env_params["useful_region"]["Default"]
        
        
        atari_env = False
        for game in atari.get_games_list():
            if game.replace("_", "") in args.env.lower():
                atari_env = True
                
        if atari_env:
            self.env = atari.make_env(self.env_name, self.env_params)
        else:
            self.env = gym.make(self.env_name, render_mode = self.env_params["render_mode"])
        self.env.metadata["render_fps"]=1000

        ## Configurar la política y parámetros del actor y del crítico
        print(self.env.observation_space)
        
        
        self.state_shape = self.env.observation_space.shape
        print("state_shape --- ", len(self.state_shape))
        
        if isinstance(self.env.action_space, gym.spaces.Discrete): #espacio discreto
            #print("Espacio discreto")
            self.action_shape = self.env.action_space.n
            self.policy = self.discrete_policy
            self.continuous_action_space = False
            
        else: #espacio de acciones continuo
            #print("action_shape ", self.env.action_space.shape)
            self.action_shape = self.env.action_space.shape[0]
            self.policy = self.multi_variate_gaussian_policy
            
        self.critic_shape = 1
        
        
        if len(self.state_shape) == 3: #Imagen de pantalla como input del agente y el crítico
            if self.continuous_action_space:# Espacio de acciones contínuas
                self.actor = DeepActor(self.state_shape, self.action_shape, device).to(device)
            else: # Espacio de acciones discretas
                self.actor = DeepDiscreteActor(self.state_shape, self.action_shape, device).to(device)
            self.critic = DeepCritic(self.state_shape, self.critic_shape, device).to(device)
        else: # Vector de cierta dimensión como input del agente y del crítico
            if self.continuous_action_space:# Espacio de acciones contínuas
                #self.actor_critic = SwallowActorCritic(slf.state_shape, self.action_shape, self.critic_shape, device).to(device)
                self.actor = SwallowActor(self.state_shape, self.action_shape, device).to(device)
            else: # Espacio de acciones discretas
                self.actor = SwallowDiscreteActor(self.state_shape, self.action_shape, device).to(device)
            self.critic = SwallowCritic(self.state_shape, self.critic_shape, device).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.params["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = self.params["learning_rate"])
        
        episode_rewards = list()
        previous_checkpoint_mean_ep_rew = self.best_mean_reward
        num_improved_episodes_before_checkpoint = 0
        
        if self.params["load_trained_model"]:
            try:
                self.load()
                previous_checkpoint_mean_ep_rew = self.best_mean_reward
            except FileNotFoundError:
                print("ERROR: No existe ningún modelo entrenado para este entorno. Empieza desde 0")
                if args.test:
                    print("FATAL: no hay modelo salvado y no podemos proceder al modo testing. Pulsa cualquier tecla para volver a empezar")
                    
                else:
                    print("WARNING: no hay ningún modelo para este entorno. Pulsa cualquier tecla para volver a empezar...")
        
        for episode in range(self.params["max_num_episodes"]):
            obs, _= self.env.reset()
            done = False
            ep_reward = 0.0
            step_num = 0
            while not done:
                action = self.get_action(obs)
                next_obs, reward, done, truncated, _ = self.env.step(action)
                self.rewards.append(reward)
                ep_reward += reward
                step_num += 1
                
                if truncated:
                    done = True

                if not args.test and (step_num > self.params["learning_step_thresh"] or done):
                    self.learn(next_obs, done)
                    step_num = 0
                    
                    if done:
                        episode_rewards.append(ep_reward)
                        if ep_reward > self.best_reward:
                            self.best_reward = ep_reward
                        
                        if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew: 
                            num_improved_episodes_before_checkpoint += 1
                        
                        if num_improved_episodes_before_checkpoint >= self.params['save_freq']:
                            previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                            self.best_mean_reward = np.mean(episode_rewards)
                            self.save()
                            num_improved_episodes_before_checkpoint = 0
                obs = next_obs
                self.global_step_num += 1
                if args.render:
                    self.env.render()
                
                if not args.test:
                    print("\n {}: Episodio #{}. ACtion = {} Con {} estados:  recompensa media = {:.2f}, mejor recompensa = {}".
                          format(self.actor_name, episode, action, ep_reward, np.mean(episode_rewards), self.best_reward))
                else:
                    print("\n {}: Episodio #{}. finalizado con una recompensa = {}".
                          format(self.actor_name, episode+1, ep_reward))

                    
                writer.add_scalar(self.actor_name+ "/reward", reward, self.global_step_num)
                writer.add_scalar(self.actor_name +"/ep_reward", ep_reward, self.global_step_num)
                writer.add_scalar(self.actor_name +"/mean_ep_reward", np.mean(episode_rewards), self.global_step_num)
                writer.add_scalar(self.actor_name +"/max_ep_reward", self.best_reward, self.global_step_num)


if __name__ == "__main__":
    
    agent_params = manager.get_agent_params()
    agent_params["model_dir"] = args.output_dir
    agent_params["test"] = args.test
    env_params = manager.get_environment_params()
    env_params["env_name"] = args.env
    
    
    
    mp.set_start_method("spawn")
    
    if args.render:
        env_params["render_mode"] ="human"
        
    agent_procs = [DeepA2CAgent(ID, args.env, agent_params, env_params) for ID in range(agent_params["num_agents"])]
    
    [p.start() for p in agent_procs]
    [p.join() for p in agent_procs]
    


    