import os
import logging
from typing import Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
import shutil

from auditor.utils.data import TestSuiteSummary


def generate_robustness_report(
    suite_summary: TestSuiteSummary,
    model_name: str,
    save_dir: str,
    logger: Optional[logging.Logger] = None,
):
    cur_dir = Path(__file__).resolve().parent
    template_dir = cur_dir / 'templates/'
    environment = Environment(loader=FileSystemLoader(template_dir))
    template = environment.get_template('report_template.html')
    info = {
        'suite_description': suite_summary.description,
        'model_name': model_name,
        'summaries': suite_summary.summaries,
    }
    rendered_template = template.render(
        info
    )
    if not os.path.isdir(save_dir):
        os.mkdir(Path(save_dir))
    filename = f"robustness_report_{model_name}.html"
    fpath = Path(save_dir) / filename
    with open(fpath, 'w', encoding='utf-8') as fid:
        fid.write(rendered_template)

    abs_path = Path(fpath).resolve()
    log_msg = f'Report generated at: {abs_path}'
    if logger is None:
        print(log_msg)
    else:
        logger.info(log_msg)
    return


def copy_css(dst_dir: str):
    css_dir = Path(__file__).parent / 'templates/css'
    try:
        dst_path = Path(dst_dir) / 'css'
        if os.path.isdir(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(css_dir, Path(dst_dir) / 'css')
    except Exception as e:
        print(f'Could not copy CSS files to destination dir: {dst_dir}')
        raise e
