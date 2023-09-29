from configs.cfg_tf import return_cfg
from trainer.trainer_tf import TrainerTf

if __name__ == '__main__':

    # make sure to provide the correct paths by modifying cfg_tf
    cfg = return_cfg(path_drop=[.9, .9])
    trainer = TrainerTf(config=cfg)

    trainer(mode='train')
