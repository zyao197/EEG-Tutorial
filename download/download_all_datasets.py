import moabb
import numpy as np
import pandas as pd
import os
import mne
import time
from moabb.datasets import BNCI2014_001, BNCI2014_002, Lee2019_MI, PhysionetMI, Schirrmeister2017

# Set MOABB data download directory
moabb.set_download_dir('./data')

# Create directories for saving processed data
save_dirs = {
    'BNCI2014_001': './data_bnci2014_001',
    'BNCI2014_002': './data_bnci2014_002',
    'Lee2019_MI': './data_lee2019_mi',
    'PhysionetMI': './data_physionet_mi',
    'Schirrmeister2017': './data_high_gamma'
}

# Create all save directories
for save_dir in save_dirs.values():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# Set bandpass filter parameters
fmin, fmax = 8, 30

# Maximum number of retries
max_retries = 3

# Define dataset configurations
dataset_configs = {
    'BNCI2014_001': {
        'class': BNCI2014_001,
        'subjects': range(1, 10),  # 9 subjects
        'event_id': {
            'left_hand': 1,
            'right_hand': 2,
            'feet': 3,
            'tongue': 4
        },
        'epoch_params': {
            'tmin': 2,
            'tmax': 6
        },
        'attrs': {
            'trial_duration': '4 seconds (2-6s)',
            'rest_period': '2-3 seconds'
        }
    },
    'BNCI2014_002': {
        'class': BNCI2014_002,
        'subjects': range(1, 15),  # 14 subjects
        'event_id': {
            'right_hand': 1,
            'feet': 2
        },
        'epoch_params': {
            'tmin': 3,
            'tmax': 8
        },
        'attrs': {
            'sampling_rate': 512,  # Hz
            'electrodes': '15 electrodes (3 Laplacian derivations at C3, Cz, C4)',
            'reference': 'left mastoid',
            'ground': 'right mastoid',
            'trial_duration': '5 seconds (3s-8s)',
            'rest_period': '2-3 seconds'
        }
    },
    'Lee2019_MI': {
        'class': Lee2019_MI,
        'subjects': range(1, 55),  # 54 subjects
        'event_id': {
            'left_hand': 1,
            'right_hand': 2
        },
        'epoch_params': {
            'tmin': 3,
            'tmax': 7
        },
        'attrs': {
            'sampling_rate': 1000,  # Hz
            'electrodes': '62 Ag/AgCl electrodes',
            'reference': 'nasion',
            'ground': 'AFz',
            'trial_duration': '4 seconds (3s-7s)',
            'rest_period': '6 seconds (±1.5s)'
        }
    },
    'PhysionetMI': {
        'class': PhysionetMI,
        'subjects': range(1, 110),  # 109 subjects
        'event_id': {
            'rest': 1,
            'left_hand': 2,
            'right_hand': 3,
            'hands': 4,
            'feet': 5
        },
        'epoch_params': {
            'tmin': 0,
            'tmax': 120  # Default value, will be adjusted based on run type
        },
        'attrs': {
            'trial_duration': '120 seconds (0s-120s)',  # Default value, will be adjusted based on run type
            'rest_period': '3-4 seconds'
        }
    },
    'Schirrmeister2017': {
        'class': Schirrmeister2017,
        'subjects': range(1, 15),  # 14 subjects
        'event_id': {
            'right_hand': 1,
            'left_hand': 2,
            'rest': 3,
            'feet': 4
        },
        'epoch_params': {
            'tmin': 0,
            'tmax': 4
        },
        'attrs': {
            'trial_duration': '4 seconds (0s-4s)',
            'rest_period': '3-4 seconds'
        }
    }
}

# Process BNCI2014_001 dataset
def process_bnci2014_001():
    dataset_name = 'BNCI2014_001'
    config = dataset_configs[dataset_name]
    save_dir = save_dirs[dataset_name]
    
    # Initialize dataset
    dataset = config['class']()
    
    # Process data for each subject
    for subject in config['subjects']:
        # Check if the subject has already been processed
        subject_files = [f for f in os.listdir(save_dir) if f.startswith(f'subject_{subject}_')]
        if subject_files:
            print(f"Subject {subject} already processed, skipping...")
            continue
            
        for retry in range(max_retries):
            try:
                print(f"Processing {dataset_name} subject {subject} (attempt {retry + 1}/{max_retries})")
                
                # Get raw data
                raw_data = dataset.get_data(subjects=[subject])
                
                # Iterate through each session
                for session in raw_data[subject].keys():
                    try:
                        # Check if the session has already been processed
                        session_file = f"subject_{subject}_session_{session}_data.csv"
                        if os.path.exists(os.path.join(save_dir, session_file)):
                            print(f"Session {session} already processed, skipping...")
                            continue
                            
                        # Get raw EEG data
                        session_data = raw_data[subject][session]
                        
                        # Iterate through each run
                        for run in session_data.keys():
                            try:
                                # Get Raw object
                                raw = session_data[run]
                                
                                # Apply bandpass filter
                                raw.filter(fmin, fmax, method='fir', phase='zero')
                                
                                # Get event information
                                events, event_dict = mne.events_from_annotations(raw)
                                
                                # Create epochs
                                epochs = mne.Epochs(raw, events, event_id=event_dict, 
                                                  tmin=config['epoch_params']['tmin'], 
                                                  tmax=config['epoch_params']['tmax'],
                                                  baseline=None, preload=True)
                                
                                # Convert to DataFrame
                                df = epochs.to_data_frame()
                                
                                # Add label encoding
                                label_map = config['event_id']
                                
                                # Add label column
                                df['label'] = df['condition'].map(label_map)
                                
                                # Rearrange columns, place label next to condition
                                cols = df.columns.tolist()
                                condition_idx = cols.index('condition')
                                cols.insert(condition_idx + 1, 'label')
                                cols.remove('label')
                                df = df[cols]
                                
                                # Add data information attributes
                                df.attrs['sampling_rate'] = raw.info['sfreq']
                                df.attrs['electrodes'] = raw.ch_names
                                df.attrs['reference'] = raw.info.get('description', 'unknown')
                                
                                # Add dataset-specific attributes
                                for key, value in config['attrs'].items():
                                    df.attrs[key] = value
                                
                                # Save as CSV file
                                filename = f"subject_{subject}_session_{session}_run_{run.split('_')[-1]}_data.csv"
                                filepath = os.path.join(save_dir, filename)
                                df.to_csv(filepath, index=False)
                                
                                print(f"Saved data for subject {subject}, session {session}, run {run.split('_')[-1]}")
                                
                            except Exception as e:
                                print(f"Error processing run {run} for subject {subject}, session {session}: {str(e)}")
                                continue
                        
                    except Exception as e:
                        print(f"Error processing session {session} for subject {subject}: {str(e)}")
                        continue
                        
                # If all sessions are successfully processed, break the retry loop
                break
                
            except Exception as e:
                print(f"Error processing subject {subject} (attempt {retry + 1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    print("Waiting 5 seconds before retrying...")
                    time.sleep(5)
                else:
                    print(f"Failed to process subject {subject} after {max_retries} attempts")
                    continue

# Process BNCI2014_002 dataset
def process_bnci2014_002():
    dataset_name = 'BNCI2014_002'
    config = dataset_configs[dataset_name]
    save_dir = save_dirs[dataset_name]
    
    # Initialize dataset
    dataset = config['class']()
    
    # Get all subjects
    subjects = dataset.subject_list
    
    # Process data for each subject
    for subject in subjects:
        print(f"Processing {dataset_name} subject {subject}")
        
        # Get raw data
        data = dataset.get_data(subjects=[subject])
        
        # Get all sessions and runs for this subject
        sessions = list(data[subject].keys())
        
        # Process each session
        for session in sessions:
            runs = list(data[subject][session].keys())
            
            for run in runs:
                # Get raw data
                raw = data[subject][session][run]
                
                # Apply bandpass filter (8-30Hz)
                raw.filter(fmin, fmax)
                
                # Get event information
                events, event_id = mne.events_from_annotations(raw)
                
                # Create epochs
                epochs = mne.Epochs(raw, events, event_id, 
                                  tmin=config['epoch_params']['tmin'], 
                                  tmax=config['epoch_params']['tmax'],
                                  baseline=None, preload=True)
                
                # Convert to DataFrame
                df = epochs.to_data_frame()
                
                # Add label encoding
                label_map = config['event_id']
                
                # Add label column
                df['label'] = df['condition'].map(label_map)
                
                # Rearrange columns, place label next to condition
                cols = df.columns.tolist()
                condition_idx = cols.index('condition')
                cols.insert(condition_idx + 1, 'label')
                cols.remove('label')
                df = df[cols]
                
                # Add data information attributes
                for key, value in config['attrs'].items():
                    df.attrs[key] = value
                
                # Save to CSV file
                output_file = os.path.join(save_dir, 
                                         f'subject_{subject}_session_{session}_run_{run}_data.csv')
                df.to_csv(output_file, index=False)
                print(f"Saved data for subject {subject}, session {session}, run {run}")

# Process Lee2019_MI dataset
def process_lee2019_mi():
    dataset_name = 'Lee2019_MI'
    config = dataset_configs[dataset_name]
    save_dir = save_dirs[dataset_name]
    
    # Initialize dataset
    dataset = config['class']()
    
    # Get all subjects
    subjects = dataset.subject_list
    
    # Process data for each subject
    for subject in subjects:
        print(f"Processing {dataset_name} subject {subject}")
        
        try:
            # Get data
            data = dataset.get_data(subjects=[subject])
            
            # Process data
            if subject in data:
                for session in data[subject].keys():
                    for run in data[subject][session].keys():
                        # Get raw data
                        raw = data[subject][session][run]
                        
                        # Apply bandpass filter (8-30Hz)
                        raw.filter(fmin, fmax)
                        
                        # Get event information
                        events, event_id = mne.events_from_annotations(raw)
                        
                        # Create epochs
                        epochs = mne.Epochs(raw, events, event_id, 
                                          tmin=config['epoch_params']['tmin'], 
                                          tmax=config['epoch_params']['tmax'],
                                          baseline=None, preload=True)
                        
                        # Convert to DataFrame
                        df = epochs.to_data_frame()
                        
                        # Add label encoding
                        label_map = config['event_id']
                        
                        # Add label column
                        df['label'] = df['condition'].map(label_map)
                        
                        # Rearrange columns, place label next to condition
                        cols = df.columns.tolist()
                        condition_idx = cols.index('condition')
                        cols.insert(condition_idx + 1, 'label')
                        cols.remove('label')
                        df = df[cols]
                        
                        # Add data information attributes
                        for key, value in config['attrs'].items():
                            df.attrs[key] = value
                        
                        # Save to CSV file
                        output_file = os.path.join(save_dir, 
                                                 f'subject_{subject}_session_{session}_run_{run}_data.csv')
                        df.to_csv(output_file, index=False)
                        print(f"Saved data for subject {subject}, session {session}, run {run}")
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            continue

# Process PhysionetMI dataset
def process_physionet_mi():
    dataset_name = 'PhysionetMI'
    config = dataset_configs[dataset_name]
    save_dir = save_dirs[dataset_name]
    
    # Initialize dataset
    dataset = config['class'](imagined=True, executed=True)  # Get both imagined and executed movement data
    
    # Define task type mapping
    task_types = {
        'imagined': {
            'hand_runs': [4, 8, 12],  # Imagined single hand movement
            'feet_runs': [6, 10, 14]  # Imagined both hands/feet movement
        },
        'executed': {
            'hand_runs': [3, 7, 11],  # Executed single hand movement
            'feet_runs': [5, 9, 13]  # Executed both hands/feet movement
        }
    }
    
    # Process data for each subject
    for subject in range(1, 110):  # 109 subjects
        # Check if the subject has already been processed
        subject_files = [f for f in os.listdir(save_dir) if f.startswith(f'subject_{subject}_')]
        if subject_files:
            print(f"Subject {subject} already processed, skipping...")
            continue
            
        for retry in range(max_retries):
            try:
                print(f"Processing {dataset_name} subject {subject} (attempt {retry + 1}/{max_retries})")
                
                # Get raw data
                raw_data = dataset.get_data(subjects=[subject])
                
                # Iterate through each session
                for session in raw_data[subject].keys():
                    try:
                        # Check if the session has already been processed
                        session_file = f"subject_{subject}_session_{session}_data.csv"
                        if os.path.exists(os.path.join(save_dir, session_file)):
                            print(f"Session {session} already processed, skipping...")
                            continue
                            
                        # Get raw EEG data
                        session_data = raw_data[subject][session]
                        
                        # Iterate through each run
                        for run in session_data.keys():
                            try:
                                # Get Raw object
                                raw = session_data[run]
                                
                                # Apply bandpass filter
                                raw.filter(fmin, fmax, method='fir', phase='zero')
                                
                                # Get event information
                                events, event_dict = mne.events_from_annotations(raw)
                                
                                # Determine task type and run type
                                run_number = int(run.split('_')[-1])
                                task_type = 'imagined' if run_number in task_types['imagined']['hand_runs'] + task_types['imagined']['feet_runs'] else 'executed'
                                run_type = 'hand' if run_number in task_types[task_type]['hand_runs'] else 'feet'
                                
                                # Create epochs (set different time windows based on task type)
                                if run_number in [1, 2]:  # Baseline recording
                                    tmin, tmax = 0, 60  # 1-minute baseline recording
                                else:
                                    tmin, tmax = 0, 120  # 2-minute task recording
                                
                                # Create epochs based on actually existing events
                                # Only use events that actually exist in the current run
                                available_events = {}
                                for event_name, event_id_value in config['event_id'].items():
                                    if event_id_value in events[:, 2]:
                                        available_events[event_name] = event_id_value
                                
                                if not available_events:
                                    print(f"No events found for subject {subject}, session {session}, run {run_number}, skipping...")
                                    continue
                                    
                                epochs = mne.Epochs(raw, events, available_events, tmin=tmin, tmax=tmax,
                                                  baseline=None, preload=True)
                                
                                # Convert to DataFrame
                                df = epochs.to_data_frame()
                                
                                # Add label encoding
                                label_map = config['event_id']
                                
                                # Add label column
                                df['label'] = df['condition'].map(label_map)
                                
                                # Rearrange columns, place label next to condition
                                cols = df.columns.tolist()
                                condition_idx = cols.index('condition')
                                cols.insert(condition_idx + 1, 'label')
                                cols.remove('label')
                                df = df[cols]
                                
                                # Add data information attributes
                                df.attrs['sampling_rate'] = raw.info['sfreq']
                                df.attrs['electrodes'] = raw.ch_names
                                df.attrs['reference'] = raw.info.get('description', 'unknown')
                                df.attrs['trial_duration'] = f"{tmax - tmin} seconds ({tmin}s-{tmax}s)"
                                df.attrs['task_type'] = task_type
                                df.attrs['run_type'] = run_type
                                df.attrs['is_baseline'] = run_number in [1, 2]
                                
                                # Save as CSV file
                                filename = f"subject_{subject}_session_{session}_run_{run_number}_data.csv"
                                filepath = os.path.join(save_dir, filename)
                                df.to_csv(filepath, index=False)
                                
                                print(f"Saved data for subject {subject}, session {session}, run {run_number}")
                                
                            except Exception as e:
                                print(f"Error processing run {run} for subject {subject}, session {session}: {str(e)}")
                                continue
                        
                    except Exception as e:
                        print(f"Error processing session {session} for subject {subject}: {str(e)}")
                        continue
                        
                # If all sessions are successfully processed, break the retry loop
                break
                
            except Exception as e:
                print(f"Error processing subject {subject} (attempt {retry + 1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    print("Waiting 5 seconds before retrying...")
                    time.sleep(5)
                else:
                    print(f"Failed to process subject {subject} after {max_retries} attempts")
                    continue

# Process Schirrmeister2017 dataset
def process_schirrmeister2017():
    dataset_name = 'Schirrmeister2017'
    config = dataset_configs[dataset_name]
    save_dir = save_dirs[dataset_name]
    
    # Initialize dataset
    dataset = config['class']()
    
    # Index of 44 sensors related to motor cortex (needs to be filled with actual channel indices based on the dataset)
    motor_cortex_channels = None  # Need to fill in the 44 motor cortex sensor indices in actual use
    
    # Process data for each subject
    for subject in range(1, 15):  # 14 subjects
        try:
            print(f"Processing {dataset_name} subject {subject}")
            
            # Get raw data
            raw_data = dataset.get_data(subjects=[subject])
            
            # Iterate through each run
            for run in range(1, 14):  # 13 runs
                try:
                    # Get raw EEG data
                    raw = raw_data[subject][1][run]  # Assume session is fixed at 1
                    
                    # If motor cortex channels are specified, only select these channels
                    if motor_cortex_channels is not None:
                        raw.pick_channels(motor_cortex_channels)
                    
                    # Apply bandpass filter
                    raw.filter(fmin, fmax, method='fir', phase='zero-phase')
                    
                    # Get event information
                    events, _ = mne.events_from_annotations(raw)
                    
                    # Create epochs
                    epochs = mne.Epochs(raw, events, config['event_id'], 
                                      tmin=config['epoch_params']['tmin'], 
                                      tmax=config['epoch_params']['tmax'],
                                      baseline=None, preload=True)
                    
                    # Convert to DataFrame
                    df = epochs.to_data_frame()
                    
                    # Add label encoding
                    label_map = config['event_id']
                    
                    # Add label column
                    df['label'] = df['condition'].map(label_map)
                    
                    # Rearrange columns, place label next to condition
                    cols = df.columns.tolist()
                    condition_idx = cols.index('condition')
                    cols.insert(condition_idx + 1, 'label')
                    cols.remove('label')
                    df = df[cols]
                    
                    # Add dataset split information (train/test)
                    is_test = run >= 12  # Last two runs as test set
                    df['is_test'] = is_test
                    
                    # Add data information attributes
                    df.attrs['sampling_rate'] = raw.info['sfreq']
                    df.attrs['electrodes'] = raw.ch_names
                    df.attrs['reference'] = raw.info.get('description', 'unknown')
                    
                    # Add dataset-specific attributes
                    for key, value in config['attrs'].items():
                        df.attrs[key] = value
                    
                    # Save as CSV file
                    set_type = 'test' if is_test else 'train'
                    filename = f"subject_{subject}_run_{run}_{set_type}_data.csv"
                    filepath = os.path.join(save_dir, filename)
                    df.to_csv(filepath, index=False)
                    
                    print(f"Saved {set_type} data for subject {subject}, run {run}")
                    
                except Exception as e:
                    print(f"Error processing run {run} for subject {subject}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing subject {subject}: {str(e)}")
            continue

# Main function
def main():
    print("Starting to download and process all datasets...")
    
    # Process BNCI2014_001 dataset
    print("\nProcessing BNCI2014_001 dataset...")
    process_bnci2014_001()
    
    # Process BNCI2014_002 dataset
    print("\nProcessing BNCI2014_002 dataset...")
    process_bnci2014_002()
    
    # Process Lee2019_MI dataset
    print("\nProcessing Lee2019_MI dataset...")
    process_lee2019_mi()
    
    # Process PhysionetMI dataset
    print("\nProcessing PhysionetMI dataset...")
    process_physionet_mi()
    
    # Process Schirrmeister2017 dataset
    print("\nProcessing Schirrmeister2017 dataset...")
    process_schirrmeister2017()
    
    print("\nAll datasets processing completed!")

if __name__ == "__main__":
    main() 