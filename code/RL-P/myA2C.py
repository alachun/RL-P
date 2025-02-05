
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from typing import Tuple, List
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from torch import nn
from typing import Any, ClassVar, Dict, Optional, Type, Union

import torch as th
from torch.distributions.categorical import Categorical
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, PyTorchObs


class MyActorCriticPolicy(ActorCriticPolicy):
    def __init__(
            self,
            env: Union[GymEnv, str],
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
            activation_fn: Type[nn.Module] = nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,

    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.env=env

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        '''这里需要修改'''
        temp=th.tensor(self.env.envs[0].env.gym_env.valid_action)
        obs_str=''.join(map(str, obs.tolist()[0]))
        self.env.envs[0].env.gym_env.masks_buffer[obs_str]=self.env.envs[0].env.gym_env.valid_action.copy()

        logits = self.action_net(latent_pi)
        action,logprob,entropy=self.get_action(logits=logits,invalid_action_masks=temp)
        return action, values, logprob

    def get_action(self, logits, action=None, invalid_action_masks=None):
        # x=logits
        split_logits = th.split(logits,tuple([int(self.env.action_space.n)]), dim=1)

        if invalid_action_masks is not None:
            if invalid_action_masks.dim()<2:
                     invalid_action_masks=invalid_action_masks.unsqueeze(0)
            split_invalid_action_masks = th.split(invalid_action_masks, tuple([int(self.env.action_space.n)]), dim=1)
            multi_categoricals = [CategoricalMasked(logits=logit, masks=iam) for (logit, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        else:
            multi_categoricals = [Categorical(logits=logit) for logit in split_logits]

        if action is None:
            action = th.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = th.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = th.stack([categorical.entropy() for categorical in multi_categoricals])
        if action.dim()>=2:
                action=action.squeeze(1)
        return action, logprob.sum(0), entropy.sum(0)



    def get_actions(self, logits, action=None, invalid_action_masks=None):
        # x=logits
        logprob_list=[]
        entropy_list=[]
        for i in range(action.shape[0]):
            logit_sample=logits[i].unsqueeze(0)
            action_sample=action[i].unsqueeze(0).unsqueeze(0)
            mask_sample=invalid_action_masks[i].unsqueeze(0)
            split_logits = th.split(logit_sample, tuple([int(self.env.action_space.n)]), dim=1)
            if mask_sample is not None:
                split_invalid_action_masks = th.split(mask_sample, tuple([int(self.env.action_space.n)]),
                                                      dim=1)
                multi_categoricals = [CategoricalMasked(logits=logit, masks=iam) for (logit, iam) in
                                      zip(split_logits, split_invalid_action_masks)]
            else:
                multi_categoricals = [Categorical(logits=logit) for logit in split_logits]

            if action_sample is None:
                action_sample = th.stack([categorical.sample() for categorical in multi_categoricals])
            logprob = th.stack([categorical.log_prob(a) for a, categorical in zip(action_sample, multi_categoricals)])
            entropy = th.stack([categorical.entropy() for categorical in multi_categoricals])
            logprob_list.append(logprob)
            entropy_list.append(entropy)
        logprobs=th.stack(logprob_list).squeeze().unsqueeze(0)
        entropys=th.stack(entropy_list).squeeze().unsqueeze(0)
        return action, logprobs, entropys





    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed

        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        temp=[]

        for i in range(obs.shape[0]):

            '''这里需要改'''
            obs_str = ''.join(map(str, obs[i].type(torch.int8).tolist()))
            valid_action_array= self.env.envs[0].env.gym_env.masks_buffer[obs_str]



            temp.append(valid_action_array)

        temp = th.BoolTensor(temp)

        #清空存储mask的字典
        self.env.envs[0].env.gym_env.masks_buffer=dict()

        logits = self.action_net(latent_pi)
        values = self.value_net(latent_vf)
        action, logprob, entropy = self.get_actions(logits=logits, action=actions,invalid_action_masks=temp)

        return values, logprob, entropy

   #重写_predict函数用于模型测试，屏蔽非法动作
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        categorical = self.get_distribution(observation)
        action_mask = self.env.envs[0].env.gym_env.valid_action
        action_mask_booltensor=th.BoolTensor(action_mask).unsqueeze(0)
        categorical.distribution= CategoricalMasked(logits=categorical.distribution.logits,
                                                                 masks=action_mask_booltensor)
        return categorical.get_actions(deterministic=deterministic)



class MyA2C(A2C):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,

    }

    def __init__(
            self,
            policy,
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 7e-4,
            n_steps: int = 5,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            rms_prop_eps: float = 1e-5,
            use_rms_prop: bool = True,
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            normalize_advantage: bool = False,
            stats_window_size: int = 100,
            tensorboard_log: Optional[str] = None,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):
        super().__init__(
            policy=MyActorCriticPolicy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rms_prop_eps=rms_prop_eps,
            use_rms_prop=use_rms_prop,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            normalize_advantage=normalize_advantage,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,#{"env":env, "observation_space":env.observation_space,"action_space":env.action_space,"lr_schedule":learning_rate},
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=True,
        )


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        print(self.lr_schedule)
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(
            self.env, self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde,
            **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)




class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(th.BoolTensor)
            #被屏蔽动作的logits设置为-1e+8
            logits = th.where(self.masks, logits, th.tensor(-1e+8))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)


    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.))
        return -p_log_p.sum(-1)



