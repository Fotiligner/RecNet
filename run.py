from litellm._logging import _disable_debugging
_disable_debugging()

import json
import sys
from logging import getLogger
import argparse
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import init_logger, get_trainer, init_seed, set_color, get_flops
from trainer import LanguageLossTrainer, SelectedUserTrainer, ITEMLanguageLossTrainer
from utils import get_model
from dataset import BPRDataset, ITEMBPRDataset


def run_baseline(model_name, dataset_name, update_item_num, kmeans_count, **kwargs):
    props = ['props/overall.yaml', f'props/{model_name}.yaml', f'props/{dataset_name}.yaml']
    # print(props)

    model_class = get_model(model_name)
    print(model_class)

    # configurations initialization
    config = Config(
        model=model_class,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=kwargs,
    )
    config['update_item_num'] = update_item_num
    config['kmeans_count'] = kmeans_count
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    # if model_name == 'BPR' or model_name == 'UUPretrain' or model_name == 'ReRec' or model_name == 'AllReRec' or model_name == 'IITest' or model_name == 'TestGames' or model_name == 'SparseReRec'\
    #         or model_name in ['TestPantry', 'TestOffice', 'IITestDiag', 'IITestDiagNew', 'TestOfficeBPR', 'TestOfficeUUPretrain','UserReRec','AgentCF', 'AgentCF_graph']:
    #     dataset = BPRDataset(config)
    # elif model_name in ['UUTest','UUTestDiag','TestOfficeUUTest']:
    #     dataset = ITEMBPRDataset(config)
    # else:
    #     dataset = create_dataset(config)

    dataset = BPRDataset(config)

    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_class(config, train_data._dataset).to(config["device"])
    logger.info(model)

    # transform = construct_transform(config)
    # flops = get_flops(model, dataset, config["device"], logger, transform)
    # logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    # if model_name in ['SASRec','BPRMF']:
    #     trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    # elif model_name in ['UUTest', 'UUTestDiag','TestOfficeUUTest','TestOfficeUUTestDiag']:
    #     trainer = ITEMLanguageLossTrainer(config,model,dataset)
    # else:
    #     trainer = LanguageLossTrainer(config,model,dataset)

    trainer = LanguageLossTrainer(config, model, dataset)

    if not config['test_only']:
        # # model training
        # trainer.fit(train_data, valid_data, saved=True, show_progress=config["show_progress"]) # 不在这里加入evaluate data
        trainer.fit(train_data, saved=True, show_progress=config["show_progress"])


    # model evaluation
    # 删除了eval data，所以当前留下来的肯定是最好的
    test_result = trainer.evaluate(test_data, model_file='./AgentCF-Sep-07-2024_16-09-29.pth', load_best_model=False, show_progress=config["show_progress"])
    # logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return model_name, dataset_name, {
        # "best_valid_score": best_valid_score,
        # "valid_score_bigger": config["valid_metric_bigger"],
        # "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="AgentCF_graph_cluster", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="CDs-100-user-dense", help="name of datasets")
    parser.add_argument("--update_item_num", "-u", type=int, default=4, help="name of datasets")
    parser.add_argument("--kmeans_count", "-k", type=int, default=10)
    args, _ = parser.parse_known_args()

    run_baseline(args.model, args.dataset, args.update_item_num, args.kmeans_count)
