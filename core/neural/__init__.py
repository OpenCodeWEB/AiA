"""AiA Neural subsystem - the self-improving brain (pure standard library).

Modules:
  - SelfEvolvingNN          : online-learning MLP with architecture evolution
  - AutoencoderCompressor   : experience compression into latent vectors
  - AnomalyDetector         : auto-threshold reconstruction-error monitoring
  - CurriculumController    : easy-to-hard task pacing scheduler
  - HopfieldMemory          : associative pattern storage & reconstruction
  - ElmanRNN                : recurrent sequence memory (truncated BPTT)
  - FederatedSync           : FedAvg weight sync across the GunX mesh
  - DQNAgent                : deep Q-learning with replay & target net
  - AttentionLayer          : scaled dot-product self-attention (learnable)
"""

from .anomaly_detector import AnomalyDetector
from .attention_layer import AttentionLayer
from .autoencoder_compressor import AutoencoderCompressor
from .curriculum_controller import CurriculumController
from .dqn_agent import DQNAgent
from .federated_sync import FederatedSync
from .hopfield_memory import HopfieldMemory
from .rnn_sequence import ElmanRNN
from .self_evolving_nn import SelfEvolvingNN

__all__ = [
    "SelfEvolvingNN",
    "AutoencoderCompressor",
    "AnomalyDetector",
    "CurriculumController",
    "HopfieldMemory",
    "ElmanRNN",
    "FederatedSync",
    "DQNAgent",
    "AttentionLayer",
]
