import pandas as pd
from utils import *
from utils import _load_csv

if __name__ == '__main__':
    question_metadata = pd.read_csv("/Users/eliferzincan/PycharmProjects/csc311_project/data/question_meta.csv")
    student_metadata = pd.read_csv("/Users/eliferzincan/PycharmProjects/csc311_project/data/student_meta.csv")
    num_missing_student = student_metadata.isna().sum()
    num_missing_question = question_metadata.isna().sum()
    print(num_missing_question)