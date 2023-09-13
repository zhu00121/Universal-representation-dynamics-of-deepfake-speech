"""
Prepare audio and metadata files for model training. The execution of this file is already included in
the train.py file.
"""
 
import os, glob
import pandas as pd
import json
from path import Path
import shutil
import logging

logger = logging.getLogger(__name__)


def prepare_data(
    metadata_train_path:str,
    metadata_valid_path:str,
    metadata_test_path:str,
    metadata_eval_path:str,
    manifest_train_path:str,
    manifest_valid_path:str,
    manifest_test_path:str,
    manifest_eval_path:str,
    ):

    # Check if this phase is already done (if so, skip it)
    if skip(manifest_train_path, manifest_valid_path, manifest_test_path, manifest_eval_path):
        logger.info("Manifest files preparation completed in previous run, skipping.")
        return
    
    if not os.path.exists(metadata_train_path):
        raise ValueError("Metadata file not found. Check the metadata folder. ")

    # List files and create manifest from list
    logger.info(
        f"Creating {manifest_train_path}, {manifest_valid_path}, {manifest_test_path}, {manifest_eval_path}"
    )
    
    # Creating json files for train, valid, and test
    create_json(metadata_train_path, manifest_train_path, 'train')
    create_json(metadata_valid_path, manifest_valid_path, 'dev')
    create_json(metadata_test_path, manifest_test_path, 'eval')
    create_json(metadata_eval_path, manifest_eval_path, 'DF_2021_eval')


def create_json(metadata_path:str, manifest_path:str, split:str):
    """
    Creates the manifest file given the metadata file.
    """
    # Load metadata file
    df_metadata = pd.read_csv(metadata_path, header=None, sep=' ')
    print(f'Total number of audio files in the current set: {df_metadata.shape[0]}')

    # Split the metadata file into train,valid,and test files
    if split != 'DF_2021_eval':
        dataframe_to_json(df_metadata,manifest_path,split)
    elif split == 'DF_2021_eval':
        DF2021_dataframe_to_json(df_metadata, manifest_path)

    logger.info(f"{manifest_path} successfully created!")


def dataframe_to_json(df,save_path,split):
    # we now build JSON examples 
    examples = {}
    for _, row in df.iterrows():
        utt_id = row.iloc[1] # e.g., LA_T_1138215
        utt_path = os.path.join(f'./data/ASVspoof_2019/LA/ASVspoof2019_LA_{split}/flac', utt_id+'.flac')
        examples[utt_id] = {"ID": utt_id,
                            "file_path": utt_path, 
                            "label": row.iloc[4]}
        
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, "w") as f:
        json.dump(examples, f, indent=4)

    return examples


def DF2021_dataframe_to_json(df,save_path):
    # we now build JSON examples 
    examples = {}
    for _, row in df.iterrows():
        utt_id = row.iloc[1] # e.g., LA_T_1138215
        utt_path = os.path.join(f'./data/ASVspoof_2021/ASVspoof2021_DF_eval/flac', utt_id+'.flac')
        examples[utt_id] = {"ID": utt_id,
                            "file_path": utt_path, 
                            "label": row.iloc[5]}
        
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, "w") as f:
        json.dump(examples, f, indent=4)

    return examples


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True