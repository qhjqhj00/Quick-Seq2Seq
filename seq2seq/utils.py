from pathlib import Path
import logging


def log_line(log):
    log.info('-' * 100)


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
