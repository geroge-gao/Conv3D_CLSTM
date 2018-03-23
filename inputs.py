#coding=utf-8
import io
import os
import sys
import math
import random
import numpy as np
from scipy.misc import imread, imresize
import logging


def load_video_list(path):
  assert os.path.exists(path)
  f = open(path, 'r')
  f_lines = f.readlines()
  f.close()
  video_data = {} 
  video_label = []
  for idx, line in enumerate(f_lines):
    video_key = '%06d' % idx
    video_data[video_key] = {}
    #从路径中将数据分割出来
    #videopath:文件路径名
    #framecnt:视频的帧数
    #video label:标签，一共有248中
    #需要将视频文件转换成图片
    videopath  = line.split(' ')[0]
    framecnt   = int(line.split(' ')[1])
    videolabel = int(line.split(' ')[2])
    video_data[video_key]['videopath'] = videopath
    video_data[video_key]['framecnt'] = framecnt
    video_label.append(videolabel)
  return video_data,video_label

def prepare_isogr_rgb_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  #fiv 向下取整函数
  scale = math.floor(div)
  '''
  scale表示视频帧和要求输入帧数的比例
  scale=0，帧数小于要求输入帧数
  scale=1，帧数大于或等于帧数，并且不超过两倍
  
  scale=0，前面的帧一次复制，不足的用最后一帧赋值
  scale=1，乘以相应的倍数，取对应的帧
  scale>1,乘相应的倍数，取对应的帧，并添加正态分布偏置。
  '''
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  #将帧数向下取整
  rand_frames = np.floor(rand_frames)

  average_values = [112,112,112]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  #产生一个[0,1]之间的数据
  crop_random = random.random()
  '''  
  随机裁剪一定的部分，然后通过resize缩放到指定尺寸
  '''
  for idx in range(0, output_frame_cnt):
    image_file = '%s%06d.jpg' %(video_path, rand_frames[idx])
    if os.path.exists(image_file)==False:
      logging.error(image_file)
      assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    '''
    对图像进行裁剪，减掉平均值？
    '''
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_isogr_depth_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      #产生一个序列[0,1,2,3,....viode_frame_cnt-1]
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(0, video_frame_cnt)
      rand_frames[video_frame_cnt::] = video_frame_cnt-1
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 0)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt-1)
  rand_frames = np.floor(rand_frames)+1

  average_values = [127,127,127] 
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in range(0, output_frame_cnt):
    image_file = '%s%06d.jpg' %(video_path, rand_frames[idx])
    assert os.path.exists(image_file)
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_skig_rgb_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 1)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt)
  rand_frames = np.floor(rand_frames)

  average_values = [132,112,96]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in range(0,output_frame_cnt):
    image_file = '%s/%04d.jpg' %(video_path, rand_frames[idx])
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images

def prepare_skig_depth_data(image_info):
  video_path = image_info[0]
  video_frame_cnt = image_info[1]
  output_frame_cnt = image_info[2]
  is_training = image_info[3]
  assert os.path.exists(video_path)
  rand_frames = np.zeros(output_frame_cnt)
  div = float(video_frame_cnt)/float(output_frame_cnt)
  scale = math.floor(div)
  if is_training:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    elif scale == 1:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt) + \
                        float(scale)/2*(np.random.random(size=output_frame_cnt)-0.5)
  else:
    if scale == 0:
      rand_frames[0:video_frame_cnt] = np.arange(1, video_frame_cnt+1)
      rand_frames[video_frame_cnt::] = video_frame_cnt
    else:
      rand_frames[::] = div*np.arange(0, output_frame_cnt)
  rand_frames[0] = max(rand_frames[0], 1)
  rand_frames[output_frame_cnt-1] = min(rand_frames[output_frame_cnt-1], video_frame_cnt)
  rand_frames = np.floor(rand_frames)

  average_values = [237,237,237]
  processed_images = np.empty((output_frame_cnt, 112, 112, 3), dtype=np.float32)
  crop_random = random.random()
  for idx in range(0,output_frame_cnt):
    image_file = '%s/%04d.jpg' %(video_path, rand_frames[idx])
    image = imread(image_file)
    image_h, image_w, image_c = np.shape(image)
    square_sz = min(image_h, image_w)
    if is_training:
      crop_h = int((image_h - square_sz)*crop_random)
      crop_w = int((image_w - square_sz)*crop_random)
    else:
      crop_h = int((image_h - square_sz)/2)
      crop_w = int((image_w - square_sz)/2)
    image_crop = image[crop_h:crop_h+square_sz,crop_w:crop_w+square_sz,::]
    processed_images[idx] = imresize(image_crop, (112,112)) - average_values
  return processed_images
