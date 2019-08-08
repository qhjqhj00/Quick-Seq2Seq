from pathlib import Path
import logging
from typing import List


def log_line(log):
    log.info('-' * 80)


def init_log(log, output_file):
    output_file.mkdir(parents=True, exist_ok=True)
    output_file = output_file / 'training.log'
    fh = logging.FileHandler(output_file, mode='a')
    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.setLevel(level=logging.INFO)


def init_output_file(base_path: Path, file_name: str) -> Path:

    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
    open(file, "w", encoding='utf-8').close()
    return file


def get_sample(total_number_of_sentences: int, percentage: float = 0.1) -> List[int]:
    """
    :param total_number_of_sentences: 数据集的大小
    :param percentage: 提取比例
    :return: sample的indices
    """
    import random
    sample_size: int = round(total_number_of_sentences * percentage)
    sample = random.sample(range(1, total_number_of_sentences), sample_size)
    return sample