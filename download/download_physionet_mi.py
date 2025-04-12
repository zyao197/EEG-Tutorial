import moabb
import numpy as np
import pandas as pd
import os
import mne
import time
from moabb.datasets import PhysionetMI

# 设置MOABB数据下载目录
moabb.set_download_dir('./data')

# 创建保存处理后数据的目录
save_dir = './data_physionet_mi'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 初始化数据集
dataset = PhysionetMI(imagined=True, executed=False)  # 只获取想象运动数据

# 设置带通滤波器参数
fmin, fmax = 8, 30

# 定义事件标签映射
event_id = {
    'rest': 1,
    'left_hand': 2,
    'right_hand': 3,
    'hands': 4,
    'feet': 5
}

# 定义任务类型映射
task_types = {
    'imagined': {
        'hand_runs': [4, 8, 12],  # 想象单手运动
        'feet_runs': [6, 10, 14]  # 想象双手/双脚运动
    }
}

# 最大重试次数
max_retries = 3

# 处理每个受试者的数据
for subject in range(1, 2):  # 109个受试者
    # 检查是否已经处理过该受试者
    subject_files = [f for f in os.listdir(save_dir) if f.startswith(f'subject_{subject}_')]
    if subject_files:
        print(f"Subject {subject} already processed, skipping...")
        continue
        
    for retry in range(max_retries):
        try:
            print(f"Processing subject {subject} (attempt {retry + 1}/{max_retries})")
            
            # 获取原始数据
            raw_data = dataset.get_data(subjects=[subject])
            
            # 获取该受试者的所有运行
            subject_data = raw_data[subject]['0']  # 根据PhysionetMI类定义，数据存储在'0'键下
            
            # 遍历每个运行
            for run_idx, run in enumerate(subject_data.keys()):
                try:
                    # 检查是否已经处理过该运行
                    run_file = f"subject_{subject}_run_{run_idx}_data.csv"
                    if os.path.exists(os.path.join(save_dir, run_file)):
                        print(f"Run {run_idx} already processed, skipping...")
                        continue
                        
                    # 获取Raw对象
                    raw = subject_data[run]
                    
                    # 只选择 EEG 通道
                    raw.pick_types(eeg=True, meg=False, stim=False, eog=False, emg=False, misc=False)
                    
                    # 应用带通滤波器
                    raw.filter(fmin, fmax, method='fir', phase='zero')
                    
                    # 获取事件信息
                    events, event_dict = mne.events_from_annotations(raw)
                    
                    # 确定运行类型和编号
                    run_number = None
                    run_type = None
                    
                    if run_idx < len(dataset.hand_runs):
                        run_type = 'hand'
                        run_number = dataset.hand_runs[run_idx]
                    else:
                        run_type = 'feet'
                        feet_idx = run_idx - len(dataset.hand_runs)
                        run_number = dataset.feet_runs[feet_idx]
                    
                    # 根据运行类型修改事件映射
                    # 原始事件映射: {'T0': 1, 'T1': 2, 'T2': 3}
                    # T0 对应 rest
                    # T1 对应 left_hand (在hand_runs中) 或 hands (在feet_runs中)
                    # T2 对应 right_hand (在hand_runs中) 或 feet (在feet_runs中)
                    
                    # 创建epochs (根据PhysionetMI类的定义，interval=[0, 3])
                    tmin, tmax = 0, 3  # 每个试验持续3秒
                    
                    # 根据运行类型和实际存在的事件创建epochs
                    available_events = {}
                    
                    # 不添加rest事件 (T0)
                    # if 1 in events[:, 2]:  # 检查是否存在T0事件
                    #     available_events['rest'] = 1
                    
                    # 根据运行类型添加其他事件
                    if run_type == 'hand':
                        # 对于hand_runs: T1对应left_hand, T2对应right_hand
                        if 2 in events[:, 2]:  # 检查是否存在T1事件
                            available_events['left_hand'] = 2
                        if 3 in events[:, 2]:  # 检查是否存在T2事件
                            available_events['right_hand'] = 3
                    else:
                        # 对于feet_runs: T1对应hands, T2对应feet
                        if 2 in events[:, 2]:  # 检查是否存在T1事件
                            available_events['hands'] = 2
                        if 3 in events[:, 2]:  # 检查是否存在T2事件
                            available_events['feet'] = 3
                    
                    if not available_events:
                        print(f"No events found for subject {subject}, run {run_number}, skipping...")
                        continue
                        
                    epochs = mne.Epochs(raw, events, available_events, tmin=tmin, tmax=tmax,
                                      baseline=None, preload=True)
                    
                    # 转换为DataFrame
                    df = epochs.to_data_frame()
                    
                    # 添加标签编码
                    label_map = {
                        'rest': 1,
                        'left_hand': 2,
                        'right_hand': 3,
                        'hands': 4,
                        'feet': 5
                    }
                    
                    # 添加label列
                    df['label'] = df['condition'].map(label_map)
                    
                    # 重新排列列，将label放在condition后面
                    cols = df.columns.tolist()
                    condition_idx = cols.index('condition')
                    cols.insert(condition_idx + 1, 'label')
                    cols.remove('label')
                    df = df[cols]
                    
                    # 添加数据信息注释
                    df.attrs['sampling_rate'] = raw.info['sfreq']
                    df.attrs['electrodes'] = raw.ch_names
                    df.attrs['reference'] = raw.info.get('description', 'unknown')
                    df.attrs['trial_duration'] = f"{tmax - tmin} seconds ({tmin}s-{tmax}s)"
                    df.attrs['run_type'] = run_type
                    df.attrs['is_baseline'] = run_number in [1, 2]
                    
                    # 保存为CSV文件
                    filename = f"subject_{subject}_run_{run_number}_data.csv"
                    filepath = os.path.join(save_dir, filename)
                    df.to_csv(filepath, index=False)
                    
                    print(f"Saved data for subject {subject}, run {run_number}")
                    
                except Exception as e:
                    print(f"Error processing run {run_idx} for subject {subject}: {str(e)}")
                    continue
                    
            # 如果成功处理完所有runs，跳出重试循环
            break
            
        except Exception as e:
            print(f"Error processing subject {subject} (attempt {retry + 1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                print("Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                print(f"Failed to process subject {subject} after {max_retries} attempts")
                continue

print("所有数据处理完成！") 