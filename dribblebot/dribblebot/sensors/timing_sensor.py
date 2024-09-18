from .sensor import Sensor
import torch
class TimingSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        # self.attached_robot_asset = attached_robot_asset

    def get_observation(self, env_ids = None):
        # print("timeing variable: ", self.env.gait_indices)
        gait_input = self.env.gait_indices.clone()
        gait_input[self.env.gait_indices > 0.5] = 0.5 - gait_input[self.env.gait_indices > 0.5]
        return torch.cat((gait_input.unsqueeze(1), -gait_input.unsqueeze(1)), dim = 1)
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(2, device=self.env.device)
    
    def get_dim(self):
        return 1