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
dataset = PhysionetMI(imagined=True, executed=True)  # 同时获取想象和实际运动数据

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
    },
    'executed': {
        'hand_runs': [3, 7, 11],  # 实际单手运动
        'feet_runs': [5, 9, 13]  # 实际双手/双脚运动
    }
}

# 最大重试次数
max_retries = 3

# 处理每个受试者的数据
for subject in range(1, 110):  # 109个受试者
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
            
            # 遍历每个session
            for session in raw_data[subject].keys():
                try:
                    # 检查是否已经处理过该session
                    session_file = f"subject_{subject}_session_{session}_data.csv"
                    if os.path.exists(os.path.join(save_dir, session_file)):
                        print(f"Session {session} already processed, skipping...")
                        continue
                        
                    # 获取原始EEG数据
                    session_data = raw_data[subject][session]
                    
                    # 遍历每个运行
                    for run in session_data.keys():
                        try:
                            # 获取Raw对象
                            raw = session_data[run]
                            
                            # 应用带通滤波器
                            raw.filter(fmin, fmax, method='fir', phase='zero')
                            
                            # 获取事件信息
                            events, event_dict = mne.events_from_annotations(raw)
                            
                            # 确定任务类型和运行类型
                            run_number = int(run.split('_')[-1])
                            task_type = 'imagined' if run_number in task_types['imagined']['hand_runs'] + task_types['imagined']['feet_runs'] else 'executed'
                            run_type = 'hand' if run_number in task_types[task_type]['hand_runs'] else 'feet'
                            
                            # 创建epochs (根据任务类型设置不同的时间窗口)
                            if run_number in [1, 2]:  # 基线记录
                                tmin, tmax = 0, 60  # 1分钟基线记录
                            else:
                                tmin, tmax = 0, 120  # 2分钟任务记录
                            
                            # 根据实际存在的事件创建epochs
                            # 只使用在当前运行中实际存在的事件
                            available_events = {}
                            for event_name, event_id_value in event_id.items():
                                if event_id_value in events[:, 2]:
                                    available_events[event_name] = event_id_value
                            
                            if not available_events:
                                print(f"No events found for subject {subject}, session {session}, run {run_number}, skipping...")
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
                            df.attrs['task_type'] = task_type
                            df.attrs['run_type'] = run_type
                            df.attrs['is_baseline'] = run_number in [1, 2]
                            
                            # 保存为CSV文件
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
                    
            # 如果成功处理完所有sessions，跳出重试循环
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