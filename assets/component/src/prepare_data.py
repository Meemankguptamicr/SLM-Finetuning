# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prepare data."""

import os
import pandas
import numpy
import sys

import argparse
import logging

import constants
from azureml.core import Run
from pathlib import Path
from raft.logconf import log_setup
from raft.raft import docTypes
from raft.raft import main as run_raft
from raft.format import main as run_format
from auth_provider import WorkspaceConnectionAuthProvider


log_setup()
logger = logging.getLogger("raft")


def get_parser_args() -> argparse.Namespace:
    """_summary_

    Returns:
        argparse.Namespace: _description_
    """
    main_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # workspace connection
    main_parser.add_argument("--chat_completion_workspace_connection", required=True, help="workspace connection for aoai model deployment")
    main_parser.add_argument("--embedding_workspace_connection", required=True, help="workspace connection for aoai model deployment")

    # input data settings
    main_parser.add_argument("--input_datapath", required=True, help="Path to the input data")
    main_parser.add_argument("--input_doctype", type=str, default="pdf", help="The type of the document, must be one of the accepted doctypes", choices=docTypes)
    main_parser.add_argument("--dataset_split_ratio", type=float, default=0.8, help="Split ratio for training and validation data")
    main_parser.add_argument("--oracle_context_percentage", type=float, default=1.0, help="The percentage that the oracle document is included in the context")

    # runtime settings
    main_parser.add_argument("--distractors", type=int, default=3, help="The number of distractor documents to include per data point / triplet")
    main_parser.add_argument("--questions", type=int, default=5, help="The number of data points / triplets to generate per chunk")
    main_parser.add_argument("--chunk_size", type=int, default=512, help="The size of each chunk in number of tokens")
    main_parser.add_argument("--qa_threshold", type=int, default=5, help="The number of Q/A samples to generate after which to stop the generation process. Defaults to -1, which means generating Q/A samples for all documents")
    main_parser.add_argument("--workers", type=int, default=2, help="The number of worker threads to use to generate the dataset")

    # output paths
    main_parser.add_argument("--generated_dataset_full", required=True, help="Output path where generated dataset will be saved in full")
    main_parser.add_argument("--generated_dataset_full_with_cot_answers", required=True, help="Output path where generated dataset will be saved in full along with cot answers")
    main_parser.add_argument("--generated_dataset_train", required=True, help="Output path where generated dataset will be saved for training")
    main_parser.add_argument("--generated_dataset_valid", required=True, help="Output path where generated dataset will be saved for validation")

    known_args, _ = main_parser.parse_known_args()
    logger.info(f"args => {known_args}")

    return known_args


def validate_env():
    """Validate OAI env vars are present in the env.

    Raises:
        Exception: if any of the requried AOAI env vars are missing.
    """
    # ensure required OAI env vars are present
    for item in constants.AZURE_OPENAI_REQUIRED_VARS:
        if os.environ.get(item, None) is None:
            raise Exception(f"key {item} does not exist in the env")


def main():
    """Run main."""
    args = get_parser_args()

    chat_completion_workspace_connection = args.chat_completion_workspace_connection
    embedding_workspace_connection = args.embedding_workspace_connection
    input_doctype = args.input_doctype
    input_datapath = Path(args.input_datapath)

    chunk_size = args.chunk_size
    distractors = args.distractors
    questions = args.questions
    qa_threshold = args.qa_threshold
    dataset_split_ratio = args.dataset_split_ratio
    oracle_context_percentage = args.oracle_context_percentage
    workers = args.workers

    generated_dataset_full = Path(args.generated_dataset_full).absolute()
    generated_dataset_train = Path(args.generated_dataset_train).absolute()
    generated_dataset_valid = Path(args.generated_dataset_valid).absolute()
    generated_dataset_full_with_cot_answers = Path(args.generated_dataset_full_with_cot_answers).absolute()

    # fetch deployment details from aoai workspace connection
    workspace_auth_provider = WorkspaceConnectionAuthProvider(connection_name=chat_completion_workspace_connection, endpoint_type=constants.EndpointType.Serverless)
    deployment_details = workspace_auth_provider.get_auth_headers()
    os.environ[constants.AZURE_OPENAI_KEY] = deployment_details['key']
    os.environ[constants.AZURE_OPENAI_ENDPOINT] = deployment_details['target']

    workspace_auth_provider = WorkspaceConnectionAuthProvider(connection_name=embedding_workspace_connection, endpoint_type=constants.EndpointType.Serverless)
    deployment_details = workspace_auth_provider.get_auth_headers()
    os.environ[constants.EMBEDDING_AZURE_OPENAI_KEY] = deployment_details['key']
    os.environ[constants.EMBEDDING_AZURE_OPENAI_ENDPOINT] = deployment_details['target']

    raft_output_dir = Path("tmp/raft/out").absolute()

    sys.argv.clear()
    sys.argv.extend([
        "raft/raft.py",
        "--datapath", input_datapath.as_posix(),
        "--doctype", input_doctype,
        "--distractors", str(distractors),
        "--questions", str(questions),
        "--chunk_size", str(chunk_size),
        "--workers", str(workers),
        "--output", raft_output_dir.as_posix(),
        "--output-format", "hf",
        "--output-type", "jsonl",
        "--p", str(oracle_context_percentage),
        # "--completion_model", "gpt-4o-mini",
        # "--embedding_model", "gpt-4o-mini",
        "--auto-clean-checkpoints", "false",
    ])

    if qa_threshold > 0:
        sys.argv.extend(["--qa-threshold", str(qa_threshold)])

    validate_env()
    run_raft()

    sys.argv.clear()
    sys.argv.extend([
        "raft/format.py",
        "--input", f"{raft_output_dir}/data-00000-of-00001.arrow",
        "--output", generated_dataset_full.as_posix(),
        "--output-format", "chat"
    ])

    run_format()

    hf_full_df = pandas.read_json(generated_dataset_full, lines=True)
    logger.info(f"total records => {len(hf_full_df)}")

    hf_train_df, hf_valid_df = numpy.split(
        hf_full_df.sample(frac=1, random_state=42),
        [int(len(hf_full_df) * dataset_split_ratio)]
    )

    logger.info(f"records in training data => {len(hf_train_df)}")
    logger.info(f"records in validation data => {len(hf_valid_df)}")

    # save dataframes to jsonl
    hf_train_df.to_json(generated_dataset_train, orient="records", lines=True)
    hf_valid_df.to_json(generated_dataset_valid, orient="records", lines=True)
    logger.info("saved training and validation dataset")

    # save raft output to the run
    run = Run.get_context()
    logger.info("Uploading raft outputs to run")
    run.upload_folder("raft_outputs", raft_output_dir)
    logger.info("Upload completed")


if __name__ == "__main__":
    main()
