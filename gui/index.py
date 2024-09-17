import streamlit as st
import json
import os


# Mocked RAFT and DSF classes for processing, replace with actual implementations
class RAFT:
    def __init__(self, pdf_path, chunk_size, total_questions, finetuning_threshold_train, finetuning_threshold_test):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.total_questions = total_questions
        self.finetuning_threshold_train = finetuning_threshold_train
        self.finetuning_threshold_test = finetuning_threshold_test

    def process(self):
        # Simulate generating JSON files for training and testing
        return {"data": "RAFT Training Data"}, {"data": "RAFT Testing Data"}


class DSF:
    def __init__(self, pdf_path, chunk_size, total_questions, finetuning_threshold_train, finetuning_threshold_test):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.total_questions = total_questions
        self.finetuning_threshold_train = finetuning_threshold_train
        self.finetuning_threshold_test = finetuning_threshold_test

    def process(self):
        # Simulate generating JSON files for training and testing
        return {"data": "DSF Training Data"}, {"data": "DSF Testing Data"}


# Streamlit App Layout
st.title("Compare RAFT vs DSF")

# Inputs
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
total_sample_dataset = st.number_input("Total Sample Dataset", min_value=1, step=1)
chunk_size = st.number_input("Chunk Size of Each Dataset", min_value=1, step=1)
total_questions_per_chunk = st.number_input("Total Number of Questions per Chunk", min_value=1, step=1)
finetuning_threshold_train = st.slider("Finetuning Threshold (Train)", 0.0, 1.0, 0.8)
finetuning_threshold_test = st.slider("Finetuning Threshold (Test)", 0.0, 1.0, 0.8)

# Process Data if input is given
if st.button("Compare RAFT vs DSF"):

    if not os.path.exists(pdf_file):
        st.error("PDF file path is invalid. Please enter a valid path.")
    else:
        # Process using RAFT
        st.write("Processing RAFT...")
        raft_model = RAFT(pdf_file, chunk_size, total_questions_per_chunk, finetuning_threshold_train,
                          finetuning_threshold_test)
        raft_train_data, raft_test_data = raft_model.process()

        # Process using DSF
        st.write("Processing DSF...")
        dsf_model = DSF(pdf_file, chunk_size, total_questions_per_chunk, finetuning_threshold_train,
                        finetuning_threshold_test)
        dsf_train_data, dsf_test_data = dsf_model.process()

        # Save output JSON files for RAFT
        with open("raft_training_data.json", "w") as train_file:
            json.dump(raft_train_data, train_file)
        with open("raft_testing_data.json", "w") as test_file:
            json.dump(raft_test_data, test_file)

        # Save output JSON files for DSF
        with open("dsf_training_data.json", "w") as train_file:
            json.dump(dsf_train_data, train_file)
        with open("dsf_testing_data.json", "w") as test_file:
            json.dump(dsf_test_data, test_file)

        # Outputs
        st.success("Comparison completed!")
        st.write("RAFT Training Data JSON Path: `raft_training_data.json`")
        st.write("RAFT Testing Data JSON Path: `raft_testing_data.json`")
        st.write("DSF Training Data JSON Path: `dsf_training_data.json`")
        st.write("DSF Testing Data JSON Path: `dsf_testing_data.json`")

        # Provide Download Links for JSON Files
        st.download_button("Download RAFT Training Data", data=json.dumps(raft_train_data),
                           file_name="raft_training_data.json")
        st.download_button("Download RAFT Testing Data", data=json.dumps(raft_test_data),
                           file_name="raft_testing_data.json")
        st.download_button("Download DSF Training Data", data=json.dumps(dsf_train_data),
                           file_name="dsf_training_data.json")
        st.download_button("Download DSF Testing Data", data=json.dumps(dsf_test_data),
                           file_name="dsf_testing_data.json")