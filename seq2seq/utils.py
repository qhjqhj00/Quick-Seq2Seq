from pathlib import Path
import logging


def log_line(log):
    """log横分隔线"""
    log.info('-' * 100)


def init_log(log, output_file):
    """初始化log设置"""
    output_file.mkdir(parents=True, exist_ok=True)
    output_file = output_file / 'training.log'
    fh = logging.FileHandler(output_file, mode='a')
    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.setLevel(level=logging.INFO)


def init_output_file(base_path: Path, file_name: str) -> Path:
    """
    在本地生成一个文件
    :param base_path: 路径
    :param file_name: 文件名
    """
    base_path.mkdir(parents=True, exist_ok=True)

    file = base_path / file_name
    open(file, "w", encoding='utf-8').close()
    return file
