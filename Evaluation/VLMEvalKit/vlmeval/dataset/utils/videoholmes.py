from ...smp import *
from .multiple_choice import extract_answer_from_item
import numpy as np
import re

FAIL_MSG = 'Failed to obtain answer via API.'

TASK_CATEGORIES = [
    'SR','IMC','TCI','TA','MHR','PAR','CTI',
]


def get_dimension_rating(data_path, score_col='score', type_col='question_type'):
    data = load(data_path)
    acc_by_type = {}
    for qtype, group in data.groupby(type_col):
        correct = (group[score_col] == 1).sum()
        total = len(group)
        acc = correct / total if total > 0 else 0
        acc_by_type[qtype] = {
            'correct': int(correct),
            'total': int(total),
            'acc': acc
        }

    total_correct = (data[score_col] == 1).sum()
    total_count = len(data)
    total_acc = total_correct / total_count if total_count > 0 else 0

    result = {
        'acc_by_type': acc_by_type,
        'total': {
            'correct': int(total_correct),
            'total': int(total_count),
            'acc': total_acc
        }
    }

    return result


# def extract_option(pred):

#     pattern = r'<answer>\s*(.*?)\s*</answer>'
#     try:
#         matches = re.findall(pattern, pred, re.DOTALL)
#     except:
#         matches = []

#     if matches:
#         choise = matches[-1].strip()
#         if 'A ' in choise or 'A:' in choise or '[A' in choise:
#             predicted_answer = 'A'
#         elif 'B ' in choise or 'B:' in choise or '[B' in choise:
#             predicted_answer = 'B'
#         elif 'C ' in choise or 'C:' in choise or '[C' in choise:
#             predicted_answer = 'C'
#         elif 'D ' in choise or 'D:' in choise or '[D' in choise:
#             predicted_answer = 'D'
#         elif 'E ' in choise or 'E:' in choise or '[E' in choise:
#             predicted_answer = 'E'
#         elif 'F ' in choise or 'F:' in choise or '[F' in choise:
#             predicted_answer = 'F'
#         elif 'A' in choise:
#             predicted_answer = 'A'
#         elif 'B' in choise:
#             predicted_answer = 'B'
#         elif 'C' in choise:
#             predicted_answer = 'C'
#         elif 'D' in choise:
#             predicted_answer = 'D'
#         elif 'E' in choise:
#             predicted_answer = 'E'
#         elif 'F' in choise:
#             predicted_answer = 'F'
#         else:
#             predicted_answer = 'WRONG'
#     else:
#         predicted_answer = 'WRONG'
#     return predicted_answer

def extract_option(pred):
    # 1. 尝试提取 <answer> 标签内的内容（若存在）
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    try:
        matches = re.findall(pattern, pred, re.DOTALL)
    except:
        matches = []

    # 如果有 <answer> 标签，就用里面的内容，否则直接用 pred
    if matches:
        choice = matches[-1].strip()
    else:
        choice = pred.strip()

    # 2. 匹配选项字母（支持 A. / A: / A、 / A ）等多种格式
    match = re.match(r'^([A-F])([\.\:、\s]|$)', choice, re.IGNORECASE)
    if match:
        predicted_answer = match.group(1).upper()
    else:
        # 兜底匹配：如果文本中含单个字母 A-F，也认为是答案
        letters = re.findall(r'\b([A-F])\b', choice, re.IGNORECASE)
        if letters:
            predicted_answer = letters[-1].upper()
        else:
            predicted_answer = 'WRONG'

    return predicted_answer