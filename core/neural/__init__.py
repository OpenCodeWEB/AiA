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
  - SpatialGridCoordinator  : (x,y,z) grid identity on the 3D neural mesh
  - ClusterHeartbeatElection: regional cluster anchor election (Y-axis)
  - SpatialTemporalSync     : distance-filtered, distance-weighted mesh sync
"""

from .anomaly_detector import AnomalyDetector
from .attention_layer import AttentionLayer
from .autoencoder_compressor import AutoencoderCompressor
from .cluster_heartbeat_election import ClusterHeartbeatElection
from .curriculum_controller import CurriculumController
from .dqn_agent import DQNAgent
from .federated_sync import FederatedSync
from .hopfield_memory import HopfieldMemory
from .rnn_sequence import ElmanRNN
from .self_evolving_nn import SelfEvolvingNN
from .spatial_grid_coordinator import SpatialGridCoordinator
from .spatial_temporal_sync import SpatialTemporalSync

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
    "SpatialGridCoordinator",
    "ClusterHeartbeatElection",
    "SpatialTemporalSync",
]
