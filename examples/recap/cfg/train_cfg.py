# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training script for CFG SFT using FSDPCfgWorker.

This script trains CFG models using a unified data loading approach that
supports both SFT data (expert trajectories) and collected data (mixed success/failure).
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.sft.fsdp_cfg_worker import FSDPCfgWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="libero_cfg_openpi")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    # GUARD (2026-07-25): README rule 1 — the advantage embedding is zero-init
    # and MUST train at SFT lr (~1e-4). At fine-tune lr (2e-6) it never moves
    # (|POS-NULL| ~0.005 vs 0.34), so conditioning is a no-op and the whole run
    # is void. This silently invalidated E032 and E040 (both launched via
    # run_cfg_sft.sh, which passes no lr override). Fail loudly instead.
    _lr = float(cfg.actor.optim.lr)
    if _lr < 1.0e-5:
        raise SystemExit(
            f"[RECAP GUARD] actor.optim.lr={_lr:g} is below 1e-5. Advantage "
            "conditioning will NOT train (README rule 1). Pass "
            "actor.optim.lr=1.0e-4 or fix the config default. "
            "Verify after training with the EMBDIAG check "
            "(|POS-NULL| should be >= 0.05)."
        )

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group using FSDPCfgWorker
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPCfgWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    runner = SFTRunner(
        cfg=cfg,
        actor=actor_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
