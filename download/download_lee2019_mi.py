import moabb
from moabb.datasets import Lee2019_MI
import numpy as np
import pandas as pd
import os
import mne

# 设置下载目录
download_path = './data_lee2019_mi'
if not os.path.exists(download_path):
    os.makedirs(download_path)

# 设置MOABB数据目录
moabb.set_download_dir('./data')

# 初始化数据集
dataset = Lee2019_MI()

# 获取所有受试者
subjects = dataset.subject_list

# 对每个受试者进行处理
for subject in subjects:
    print(f"Processing subject {subject}")
    
    try:
        # 获取数据
        data = dataset.get_data(subjects=[subject])
        
        # 处理数据
        if subject in data:
            for session in data[subject].keys():
                for run in data[subject][session].keys():
                    # 获取原始数据
                    raw = data[subject][session][run]
                    
                    # 应用带通滤波器 (8-30Hz)
                    raw.filter(8, 30)
                    
                    # 获取事件信息
                    events, event_id = mne.events_from_annotations(raw)
                    
                    # 创建epochs (从提示出现后开始，持续4秒)
                    epochs = mne.Epochs(raw, events, event_id, tmin=3, tmax=7,
                                      baseline=None, preload=True)
                    
                    # 转换为DataFrame
                    df = epochs.to_data_frame()
                    
                    # 添加标签编码
                    label_map = {
                        'left_hand': 1,
                        'right_hand': 2
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
                    df.attrs['sampling_rate'] = 1000  # Hz
                    df.attrs['electrodes'] = '62 Ag/AgCl electrodes'
                    df.attrs['reference'] = 'nasion'
                    df.attrs['ground'] = 'AFz'
                    df.attrs['trial_duration'] = '4 seconds (3s-7s)'
                    df.attrs['rest_period'] = '6 seconds (±1.5s)'
                    
                    # 保存到CSV文件
                    output_file = os.path.join(download_path, 
                                             f'subject_{subject}_session_{session}_run_{run}_data.csv')
                    df.to_csv(output_file, index=False)
                    print(f"Saved data for subject {subject}, session {session}, run {run}")
    except Exception as e:
        print(f"Error processing subject {subject}: {str(e)}")
        continue

print("所有数据处理完成！") 