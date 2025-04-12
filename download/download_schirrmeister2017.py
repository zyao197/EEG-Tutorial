import moabb
import numpy as np
import pandas as pd
import os
import mne
from moabb.datasets import Schirrmeister2017

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 设置MOABB数据下载目录
moabb.set_download_dir('./data')

# 创建保存处理后数据的目录
save_dir = './data_high_gamma'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 初始化数据集
dataset = Schirrmeister2017()

# 运动皮层相关的44个传感器的索引（这里需要根据实际数据集确定具体的通道）
motor_cortex_channels = None  # 在实际使用时需要填入44个运动皮层传感器的索引

# 设置带通滤波器参数
fmin, fmax = 8, 30

# 定义事件标签映射
event_id = {
    'right_hand': 1,
    'left_hand': 2,
    'rest': 3,
    'feet': 4
}

# 处理每个受试者的数据
for subject in range(1, 2):  # 14个受试者
    try:
        print(f"Processing subject {subject}")
        
        # 获取原始数据
        raw_data = dataset.get_data(subjects=[subject])
        
        # 遍历每个run
        for run in range(1, 3):  # 13个runs
            try:
                # 获取原始EEG数据
                raw = raw_data[subject][1][run]  # 假设session固定为1
                
                # 只选择 EEG 通道
                raw.pick_types(eeg=True, meg=False, stim=False, eog=False, emg=False, misc=False)
                
                # 如果指定了运动皮层通道，则只选择这些通道
                if motor_cortex_channels is not None:
                    raw.pick_channels(motor_cortex_channels)
                
                # 应用带通滤波器
                raw.filter(fmin, fmax, method='fir', phase='zero-phase')
                
                # 获取事件信息
                events, _ = mne.events_from_annotations(raw)
                
                # 创建epochs (4秒trials)
                epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4,
                                  baseline=None, preload=True)
                
                # 转换为DataFrame
                df = epochs.to_data_frame()
                
                # 添加标签编码
                label_map = {
                    'right_hand': 1,
                    'left_hand': 2,
                    'rest': 3,
                    'feet': 4
                }
                
                # 添加label列
                df['label'] = df['condition'].map(label_map)
                
                # 重新排列列，将label放在condition后面
                cols = df.columns.tolist()
                condition_idx = cols.index('condition')
                cols.insert(condition_idx + 1, 'label')
                cols.remove('label')
                df = df[cols]
                
                # 添加数据集划分信息（训练/测试）
                is_test = run >= 12  # 最后两个runs作为测试集
                df['is_test'] = is_test
                
                # 添加数据信息注释
                df.attrs['sampling_rate'] = raw.info['sfreq']
                df.attrs['electrodes'] = raw.ch_names
                df.attrs['reference'] = raw.info.get('description', 'unknown')
                df.attrs['trial_duration'] = '4 seconds (0s-4s)'
                df.attrs['rest_period'] = '3-4 seconds'
                
                # 保存为CSV文件
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

print("所有数据处理完成！") 