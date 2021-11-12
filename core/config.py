import os
import numpy as np


class Config(object):
    def __init__(self):
        self.ai_catalog = 'YN_ai_1030.txt'
        self.man_catalog = 'YN-man-1030.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [25.50, 25.80]
        self.lon_range = [99.80, 100.05]
        # Boundary could be 'city','county','province' and 'N'
        self.boundary = 'N'
        self.time_range = ['20211030T000000.001', '20211031T000000.001']
        self.main_eq_starttime = '20211030T000000.001'
        # If there is no main earthquake , you can set 'self.main_eq_starttime = self.time_range[0]'
        self.plot_lat = [25.5, 25.76]
        self.plot_lon = [99.7, 100.22]
        self.ground_truth = 0.5
        self.cpu_cores = 120


class Config_SC(object):
    def __init__(self):
        self.ai_catalog = '20210609-20210612risp.log'
        self.man_catalog = '20210609-0612rengong.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [26, 34]
        self.lon_range = [97, 107]
        self.time_range = ['20210609T000000.001', '20210613T000000.001']
        self.main_eq_starttime = self.time_range[0]
        # If there is no main earthquake , you can set 'self.main_eq_starttime = self.time_range[0]'
        self.plot_lat = [26, 34]
        self.plot_lon = [97, 107]
        self.ground_truth = 0.5
        self.cpu_cores = 120


class Config_FJ(object):
    def __init__(self):
        self.ai_catalog = 'Associate_All_FJ_2021_0101_0131.txt'
        self.man_catalog = 'BulletinPrint_All_FJ_2021_0101_0131.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [21, 27]
        self.lon_range = [114, 122]
        self.time_range = ['20210101T000000.001', '20210201T000000.001']
        self.main_eq_starttime = self.time_range[0]
        # If there is no main earthquake , you can set 'self.main_eq_starttime = self.time_range[0]'
        self.plot_lat = [21, 27]
        self.plot_lon = [114, 122]
        self.ground_truth = 0.5
        self.cpu_cores = 120

class Config_QH(object):
    def __init__(self):
        self.ai_catalog = 'QH_AI.log'
        self.man_catalog = 'QH_man.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [31.7, 39.4]
        self.lon_range = [89, 104]
        self.time_range = ['20210801T000000.001', '20210901T000000.001']
        self.main_eq_starttime = self.time_range[0]
        # self.main_eq_starttime = '20210801T000000.001'
        # If there is no main earthquake , you can set 'self.main_eq_starttime = self.time_range[0]'
        self.plot_lat = [31.7,39.4]
        self.plot_lon = [89, 104]
        self.ground_truth = 0.5
        self.cpu_cores = 120


class Config_SC2(object):
    def __init__(self):
        self.ai_catalog = 'SC_AI2.log'
        self.man_catalog = 'SC_man2.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [26, 34]
        self.lon_range = [97, 107]
        self.time_range = ['20210609T000000.001', '20210613T000000.001']
        # Boundary could be 'city','county','province' and 'N'
        self.boundary = 'province'
        self.main_eq_starttime = self.time_range[0]
        # If there is no main earthquake , you can set 'self.main_eq_starttime = self.time_range[0]'
        self.plot_lat = [26, 34]
        self.plot_lon = [97, 107]
        self.ground_truth = 0.5
        self.cpu_cores = 120