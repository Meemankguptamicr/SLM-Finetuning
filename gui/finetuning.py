class Base:
    def __init__(self, pdf_path, chunk_size, total_questions, finetuning_threshold_train, finetuning_threshold_test):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.total_questions = total_questions
        self.finetuning_threshold_train = finetuning_threshold_train
        self.finetuning_threshold_test = finetuning_threshold_test


class RAFT(Base):
    def __init__(self, pdf_path, chunk_size, total_questions, finetuning_threshold_train, finetuning_threshold_test):
        super().__init__(pdf_path, chunk_size, total_questions, finetuning_threshold_train, finetuning_threshold_test)

    def process(self):
        # todo integrate with pipeline
        return {"data": "RAFT Training Data"}, {"data": "RAFT Testing Data"}


class DSF(Base):
    def __init__(self, pdf_path, chunk_size, total_questions, finetuning_threshold_train, finetuning_threshold_test):
        super().__init__(pdf_path, chunk_size, total_questions, finetuning_threshold_train, finetuning_threshold_test)

    def process(self):
        # todo integrate with pipeline
        return {"data": "DSF Training Data"}, {"data": "DSF Testing Data"}
