@dataclass
class Params:
  normalize_advantage: bool = True
  policy: str = "MlpPolicy"
  batch_size: int = 256
  n_steps: int = 32
  gamma: float = 0.9999
  learning_rate: float = float(7.77e-05)
  ent_coef: float = 0.00429
  clip_range: float = 0.1
  n_epochs: int = 100
  gae_lambda: float = 0.9
  max_grad_norm: int = 5
  vf_coef: float = 0.19
  use_sde: bool = True
  policy_kwargs: Dict = field(default_factory=lambda: dict(log_std_init=-3.29, ortho_init=False))