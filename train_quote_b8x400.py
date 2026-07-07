from functools import partial

import numpy as np
import yaml

from src.data_generation.data_preperation import grid_from_cfg
from src.data_generation.noise import make_quote_eval_set, quote_data_preparation
from src.model.finetune import finetune
from src.model.quote_loss import quote_arb_loss

RUN_NAME = "ssvi_quote_uniform_3_60_b8x400"
N_HELDOUT = 15
GROUP_SIZE = 4
VAL_SEED = 0 

cfg = yaml.safe_load(open("config.yaml"))
ttms, ks = grid_from_cfg(cfg)

data_provider = partial(quote_data_preparation, cfg, n_context=(3, 60),
                        n_heldout=N_HELDOUT, size_group=GROUP_SIZE)
loss_fn = partial(quote_arb_loss, grid_shape=(len(ttms), len(ks)), lambda_cal=1.0, lambda_bf=1.0)

np.random.seed(VAL_SEED)
val_sets = [make_quote_eval_set(cfg, 8, s, N_HELDOUT) for s in (5, 10, 20, 40, 60)]
val_data = (sum((v[0] for v in val_sets), []), sum((v[1] for v in val_sets), []))

finetune(data_provider, run_name=RUN_NAME, n_epochs=300, n_surfaces_per_epoch=400,
         batch_size=2 * GROUP_SIZE, group_size=GROUP_SIZE, val_every=5,
         val_data=val_data, loss_fn=loss_fn, device="cuda")
