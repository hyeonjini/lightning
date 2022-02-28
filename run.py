import dotenv
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):

    from src import utils
    from src.training_pipeline import train

    utils.extras(config)

    return train(config)


if __name__ == "__main__":
    main()
    