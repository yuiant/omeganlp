#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from omeganlp.task import Task


@hydra.main(config_path="./configs", config_name="config.yaml")
def main(config: DictConfig):

    seed_everything(config.seed)

    task = Task.load(config)
    task.train()
    task.test()
    # task.save_inferer()
    print("finish")


if __name__ == "__main__":
    main()
