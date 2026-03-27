'''
Author: danielwangow daomiao.wang@live.com
Date: 2025-09-03 18:20:42
LastEditors: danielwangow daomiao.wang@live.com
LastEditTime: 2025-09-05 18:44:12
FilePath: /TDA-Homology/utils/dataTransfer.py
Description: Data segmentation utilities for time series data
-----> VENI VIDI VICI <-----
Copyright (c) 2025 by Daniel.Wang@Fudan University. All Rights Reserved.
'''
import os
import numpy as np
import pandas as pd
import json
from scipy import stats


def segment_and_save_data(file_path: str, output_path: str):
    """
    根据文件路径读取数据，按照分割要求分割数据，并保存为PKL-JSON格式
    
    Args:
        file_path: 输入数据文件路径 (格式: data,label)
        output_path: 输出PKL文件路径
    """
    # 读取数据文件
    df = pd.read_csv(file_path, header=None).to_numpy()
    data = df[:, 0].astype(float)
    label = df[:, 1].astype(int)
    
    # 找到标签变化的位置
    transitions = np.where(label[1:] != label[:-1])[0] + 1
    
    # 找到异常段
    anomaly_segments = []
    if len(transitions) > 0:
        if label[0] == 1:  # 开始就是异常
            for i in range(0, len(transitions), 2):
                start = 0 if i == 0 else transitions[i]
                end = transitions[i] if i == 0 else (transitions[i+1] if i+1 < len(transitions) else len(label))
                anomaly_segments.append((start, end))
        else:  # 开始是正常
            for i in range(0, len(transitions), 2):
                start = transitions[i]
                end = transitions[i+1] if i+1 < len(transitions) else len(label)
                anomaly_segments.append((start, end))
    
    # 计算分割点 (异常段之间的中点)
    split_points = []
    for i in range(len(anomaly_segments) - 1):
        current_end = anomaly_segments[i][1]
        next_start = anomaly_segments[i+1][0]
        split_point = (current_end + next_start) // 2
        split_points.append(split_point)
    
    # 分割数据
    data_segments = []
    label_segments = []
    start_idx = 0
    
    for split_point in split_points:
        data_segments.append(data[start_idx:split_point])
        label_segments.append(label[start_idx:split_point])
        start_idx = split_point
    
    # 添加最后一段
    data_segments.append(data[start_idx:])
    label_segments.append(label[start_idx:])
    
    # 如果没有找到多个异常段，就简单分割为较小的片段
    if len(data_segments) <= 1:
        # 简单分割为固定长度的段
        segment_length = 1000
        data_segments = []
        label_segments = []
        
        for i in range(0, len(data), segment_length):
            end_idx = min(i + segment_length, len(data))
            data_segments.append(data[i:end_idx])
            label_segments.append(label[i:end_idx])
    
    # 准备PKL和JSON数据
    pkl_data = []
    metadata = []
    
    for idx, (data_seg, label_seg) in enumerate(zip(data_segments, label_segments)):
        if len(data_seg) == 0:
            continue
        
        # 确定类别
        clz = 1 if np.any(label_seg != 0) else 0
        
        # 找到此段内的异常位置（相对于当前段的位置）
        signature_locations = []
        seg_transitions = np.where(label_seg[1:] != label_seg[:-1])[0] + 1
        
        if len(seg_transitions) > 0:
            if label_seg[0] == 1:  # 段开始就是异常
                for i in range(0, len(seg_transitions), 2):
                    start = 0 if i == 0 else seg_transitions[i]
                    end = seg_transitions[i] if i == 0 else (seg_transitions[i+1] if i+1 < len(seg_transitions) else len(label_seg))
                    signature_locations.append([int(start), int(end)])
            else:  # 段开始是正常
                for i in range(0, len(seg_transitions), 2):
                    start = seg_transitions[i]
                    end = seg_transitions[i+1] if i+1 < len(seg_transitions) else len(label_seg)
                    signature_locations.append([int(start), int(end)])
        elif np.all(label_seg == 1):  # 整段都是异常
            signature_locations.append([0, int(len(label_seg))])
        
        # 计算统计参数
        mean_val = np.mean(data_seg)
        skewness = stats.skew(data_seg)
        mae = np.mean(np.abs(data_seg - mean_val))
        mse = np.mean((data_seg - mean_val) ** 2)
        
        signature_params = {
            "len": int(len(data_seg)),
            "skew": float(skewness),
            "mae": float(mae),
            "mse": float(mse)
        }
        
        # 添加到PKL数据（保持原始长度）
        pkl_data.append({
            'data': data_seg.tolist()
        })
        
        # 添加到元数据
        metadata_entry = {
            "signature_params": signature_params,
            "signature_locations": signature_locations
        }
        metadata.append(metadata_entry)
    
    os.makedirs(output_path, exist_ok=True)

    output_name = file_path.split('.')[0].split('/')[-1] + "_data.pkl"
    output_metadata_name = file_path.split('.')[0].split('/')[-1] + "_metadata.json"
    output_pkl_path = os.path.join(output_path, output_name)
    output_metadata_path = os.path.join(output_path, output_metadata_name)
    
    # 保存PKL文件
    pd.to_pickle(pkl_data, output_pkl_path)

    # 保存JSON元数据
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"分割完成: {len(data_segments)} 个数据段")
    print(f"PKL文件保存至: {output_pkl_path}")
    
    return pkl_data, metadata


def batch_segment_and_save_data(file_paths: list, output_path: str, output_name: str = "batch_data"):
    """
    批量处理多个文件的数据分割，并合并存储在一对PKL-JSON文件中
    
    Args:
        file_paths: 输入数据文件路径列表
        output_path: 输出目录路径
        output_name: 输出文件名前缀（默认为"batch_data"）
    
    Returns:
        all_pkl_data: 所有文件的数据段列表
        all_metadata: 所有文件的元数据列表
    """
    all_pkl_data = []
    all_metadata = []
    
    # 遍历每个文件
    for file_idx, file_path in enumerate(file_paths):
        print(f"\n处理文件 {file_idx + 1}/{len(file_paths)}: {file_path}")
        
        try:
            # 读取数据文件
            df = pd.read_csv(file_path, header=None).to_numpy()
            data = df[:, 0].astype(float)
            label = df[:, 1].astype(int)
            
            # 找到标签变化的位置
            transitions = np.where(label[1:] != label[:-1])[0] + 1
            
            # 找到异常段
            anomaly_segments = []
            if len(transitions) > 0:
                if label[0] == 1:  # 开始就是异常
                    for i in range(0, len(transitions), 2):
                        start = 0 if i == 0 else transitions[i]
                        end = transitions[i] if i == 0 else (transitions[i+1] if i+1 < len(transitions) else len(label))
                        anomaly_segments.append((start, end))
                else:  # 开始是正常
                    for i in range(0, len(transitions), 2):
                        start = transitions[i]
                        end = transitions[i+1] if i+1 < len(transitions) else len(label)
                        anomaly_segments.append((start, end))
            
            # 计算分割点 (异常段之间的中点)
            split_points = []
            for i in range(len(anomaly_segments) - 1):
                current_end = anomaly_segments[i][1]
                next_start = anomaly_segments[i+1][0]
                split_point = (current_end + next_start) // 2
                split_points.append(split_point)
            
            # 分割数据
            data_segments = []
            label_segments = []
            start_idx = 0
            
            for split_point in split_points:
                data_segments.append(data[start_idx:split_point])
                label_segments.append(label[start_idx:split_point])
                start_idx = split_point
            
            # 添加最后一段
            data_segments.append(data[start_idx:])
            label_segments.append(label[start_idx:])
            
            # 如果没有找到多个异常段，就简单分割为较小的片段
            if len(data_segments) <= 1:
                # 简单分割为固定长度的段
                segment_length = 1000
                data_segments = []
                label_segments = []
                
                for i in range(0, len(data), segment_length):
                    end_idx = min(i + segment_length, len(data))
                    data_segments.append(data[i:end_idx])
                    label_segments.append(label[i:end_idx])
            
            # 处理每个数据段
            for seg_idx, (data_seg, label_seg) in enumerate(zip(data_segments, label_segments)):
                if len(data_seg) == 0:
                    continue
                
                # 确定类别
                clz = 1 if np.any(label_seg != 0) else 0
                
                # 找到此段内的异常位置（相对于当前段的位置）
                signature_locations = []
                seg_transitions = np.where(label_seg[1:] != label_seg[:-1])[0] + 1
                
                if len(seg_transitions) > 0:
                    if label_seg[0] == 1:  # 段开始就是异常
                        for i in range(0, len(seg_transitions), 2):
                            start = 0 if i == 0 else seg_transitions[i]
                            end = seg_transitions[i] if i == 0 else (seg_transitions[i+1] if i+1 < len(seg_transitions) else len(label_seg))
                            signature_locations.append([int(start), int(end)])
                    else:  # 段开始是正常
                        for i in range(0, len(seg_transitions), 2):
                            start = seg_transitions[i]
                            end = seg_transitions[i+1] if i+1 < len(seg_transitions) else len(label_seg)
                            signature_locations.append([int(start), int(end)])
                elif np.all(label_seg == 1):  # 整段都是异常
                    signature_locations.append([0, int(len(label_seg))])
                
                # 计算统计参数
                mean_val = np.mean(data_seg)
                skewness = stats.skew(data_seg)
                mae = np.mean(np.abs(data_seg - mean_val))
                mse = np.mean((data_seg - mean_val) ** 2)
                
                signature_params = {
                    "len": int(len(data_seg)),
                    "skew": float(skewness),
                    "mae": float(mae),
                    "mse": float(mse)
                }
                
                # 添加到PKL数据（保持原始长度）
                pkl_entry = {
                    'data': data_seg.tolist()
                }
                all_pkl_data.append(pkl_entry)
                
                # 添加到元数据
                metadata_entry = {
                    "signature_params": signature_params,
                    "signature_locations": signature_locations
                }
                all_metadata.append(metadata_entry)
            
            print(f"文件 {file_path} 处理完成: {len(data_segments)} 个数据段")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 定义输出文件路径
    output_pkl_path = os.path.join(output_path, f"{output_name}.pkl")
    output_metadata_path = os.path.join(output_path, f"{output_name}_metadata.json")
    
    # 保存PKL文件
    pd.to_pickle(all_pkl_data, output_pkl_path)
    
    # 保存JSON元数据
    with open(output_metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n批量处理完成:")
    print(f"总共处理: {len(file_paths)} 个文件")
    print(f"总共生成: {len(all_pkl_data)} 个数据段")
    print(f"PKL文件保存至: {output_pkl_path}")
    print(f"元数据保存至: {output_metadata_path}")
    
    return all_pkl_data, all_metadata

