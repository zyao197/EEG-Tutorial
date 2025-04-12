import moabb
from moabb.datasets import BNCI2014_002
import numpy as np
import pandas as pd
import os
import mne

# 设置下载目录
download_path = './data_bnci2014_002'
if not os.path.exists(download_path):
    os.makedirs(download_path)

# 初始化数据集
dataset = BNCI2014_002()

# 获取所有受试者
subjects = dataset.subject_list

# 对每个受试者进行处理
for subject in subjects:
    print(f"Processing subject {subject}")
    
    # 获取原始数据
    data = dataset.get_data(subjects=[subject])
    
    # 获取该受试者的所有会话和运行
    sessions = list(data[subject].keys())
    
    # 对每个会话进行处理
    for session in sessions:
        runs = list(data[subject][session].keys())
        
        for run in runs:
            # 获取原始数据
            raw = data[subject][session][run]
            
            # 只选择 EEG 通道
            raw.pick_types(eeg=True, meg=False, stim=False, eog=False, emg=False, misc=False)
            
            # 应用带通滤波器 (8-30Hz)
            raw.filter(8, 30)
            
            # 获取事件信息
            events, event_id = mne.events_from_annotations(raw)
            
            # 创建epochs (从3s到8s，这是实际的MI执行时间)
            epochs = mne.Epochs(raw, events, event_id, tmin=3, tmax=8,
                              baseline=None, preload=True)
            
            # 转换为DataFrame
            df = epochs.to_data_frame()
            
            # 添加标签编码（根据数据集描述，事件编码为：right_hand: 1, feet: 2）
            label_map = {
                'right_hand': 1,
                'feet': 2
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
            df.attrs['sampling_rate'] = 512  # Hz
            df.attrs['electrodes'] = '15 electrodes (3 Laplacian derivations at C3, Cz, C4)'
            df.attrs['reference'] = 'left mastoid'
            df.attrs['ground'] = 'right mastoid'
            df.attrs['trial_duration'] = '5 seconds (3s-8s)'
            df.attrs['rest_period'] = '2-3 seconds'
            
            # 保存到CSV文件
            output_file = os.path.join(download_path, 
                                     f'subject_{subject}_session_{session}_run_{run}_data.csv')
            df.to_csv(output_file, index=False)
            print(f"Saved data for subject {subject}, session {session}, run {run}")

print("所有数据处理完成！") 