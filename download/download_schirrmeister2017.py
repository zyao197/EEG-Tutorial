import moabb
import numpy as np
import pandas as pd
import os
import mne
from moabb.datasets import Schirrmeister2017
import traceback

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

def process_raw_data(raw_data, is_test=False, subject=None):
    """处理原始数据并返回DataFrame"""
    try:
        # 选择EEG通道
        raw_data.pick(picks=['eeg'], exclude='bads')
        
        # 如果指定了运动皮层通道，则只选择这些通道
        if motor_cortex_channels is not None:
            raw_data.pick_channels(motor_cortex_channels)
        
        # 应用带通滤波器
        raw_data.filter(fmin, fmax, method='fir', phase='zero')
        
        # 获取事件信息
        try:
            events, event_dict = mne.events_from_annotations(raw_data)
        except ValueError:
            events = np.array([[0, 0, 1]])
        
        # 创建epochs (4秒trials)
        epochs = mne.Epochs(raw_data, events, event_id, tmin=0, tmax=4,
                          baseline=None, preload=True)
        
        # 转换为DataFrame
        df = epochs.to_data_frame()
        
        # 添加标签编码
        df['label'] = df['condition'].map(event_id)
        
        # 重新排列列，将label放在condition后面
        cols = df.columns.tolist()
        condition_idx = cols.index('condition')
        cols.insert(condition_idx + 1, 'label')
        cols.remove('label')
        df = df[cols]
        
        # 添加数据集划分信息
        df['is_test'] = is_test
        
        # 添加数据信息注释
        df.attrs['sampling_rate'] = raw_data.info['sfreq']
        df.attrs['electrodes'] = raw_data.ch_names
        df.attrs['reference'] = raw_data.info.get('description', 'unknown')
        df.attrs['trial_duration'] = '4 seconds (0s-4s)'
        df.attrs['rest_period'] = '3-4 seconds'
        
        return df
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        return None

def save_dataframe(df, subject, is_test):
    """保存DataFrame到CSV文件"""
    if df is None:
        return
    
    data_type = "test" if is_test else "train"
    filename = f"subject_{subject}_{data_type}_data.csv"
    filepath = os.path.join(save_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"已保存{data_type}数据到: {filepath}")

# 处理每个受试者的数据
for subject in range(1, 15):  # 14个受试者
    try:
        print(f"\n开始处理受试者 {subject}")
        
        # 获取原始数据
        raw_data = dataset.get_data(subjects=[subject])
        
        # 获取训练集和测试集数据
        sessions = raw_data[subject]
        train_raw = sessions['0']['0train']  # 训练数据
        test_raw = sessions['0']['1test']    # 测试数据
        
        # 处理训练数据
        print("处理训练数据...")
        train_df = process_raw_data(train_raw, is_test=False, subject=subject)
        save_dataframe(train_df, subject, is_test=False)
        
        # 处理测试数据
        print("处理测试数据...")
        test_df = process_raw_data(test_raw, is_test=True, subject=subject)
        save_dataframe(test_df, subject, is_test=True)
                
    except Exception as e:
        print(f"处理受试者 {subject} 时出错: {str(e)}")
        continue

print("\n所有数据处理完成！") 