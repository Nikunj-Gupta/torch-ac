import numpy 
import torch
import torch.nn.functional as F 
torch.autograd.set_detect_anomaly(True)

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size 

        assert self.batch_size % self.recurrence == 0 

        self.parameter_set1 = [param for name, param in self.acmodel.named_parameters() if 'hammer' not in name and 'voi' not in name] 
        self.parameter_set2 = [param for name, param in self.acmodel.named_parameters() if 'hammer' in name or 'voi' in name]  

        self.optimizer = torch.optim.Adam(self.parameter_set1, lr, eps=adam_eps)
        self.optimizer2 = torch.optim.Adam(self.parameter_set2, lr, eps=adam_eps)

        self.batch_num = 0

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_costs = []
            log_inquiries = []
            log_policy_losses = []
            log_value_losses = []
            log_policy_losses2 = []
            log_value_losses2 = []
            log_grad_norms = []
            log_grad_norms2 = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_costs = 0
                batch_inquiries = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_policy_loss2 = 0
                batch_value_loss2 = 0
                batch_loss = 0
                batch_loss2 = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, cost, ask, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value, cost, ask = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    surr1_2 = ratio * sb.advantage2
                    surr2_2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage2
                    policy_loss2 = -torch.min(surr1_2, surr2_2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()
                    
                    surr1_2 = (value - sb.returnn2).pow(2)
                    surr2_2 = (value_clipped - sb.returnn2).pow(2)
                    value_loss2 = torch.max(surr1_2, surr2_2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                    loss2 = policy_loss2 - self.entropy_coef * entropy + self.value_loss_coef * value_loss2

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_costs += cost.mean().item()
                    batch_inquiries += ask.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_policy_loss2 += policy_loss2.item()
                    batch_value_loss2 += value_loss2.item()
                    batch_loss += loss
                    batch_loss2 += loss2

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_costs /= self.recurrence
                batch_inquiries /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_policy_loss2 /= self.recurrence
                batch_value_loss2 /= self.recurrence
                batch_loss /= self.recurrence
                batch_loss2 /= self.recurrence

                # Update actor-critic

                
                
                self.optimizer.zero_grad()
                batch_loss.backward(retain_graph=True)
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.parameter_set1) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.parameter_set1, self.max_grad_norm)
                

            
                self.optimizer2.zero_grad()
                batch_loss2.backward()
                grad_norm2 = sum(p.grad.data.norm(2).item() ** 2 for p in self.parameter_set2) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.parameter_set2, self.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer2.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_costs.append(batch_costs)
                log_inquiries.append(batch_inquiries)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_policy_losses2.append(batch_policy_loss2)
                log_value_losses2.append(batch_value_loss2)
                log_grad_norms.append(grad_norm)
                log_grad_norms2.append(grad_norm2)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "cost": numpy.mean(log_costs),
            "inquiries": numpy.mean(log_inquiries),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "policy_loss2": numpy.mean(log_policy_losses2),
            "value_loss2": numpy.mean(log_value_losses2),
            "grad_norm": numpy.mean(log_grad_norms), 
            "grad_norm2": numpy.mean(log_grad_norms2)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
