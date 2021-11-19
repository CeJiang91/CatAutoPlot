class Config(object):
    def __init__(self):
        self.ai_catalog = 'YN_ai_1030.txt'
        self.man_catalog = 'YN-man-1030.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [21, 30]
        self.lon_range = [95, 108]
        # Boundary could be 'city','county','province' and 'N'
        self.boundary = 'province'
        self.time_range = ['20211030T000000.001', '20211031T000000.001']
        self.main_eq_starttime = '20211030T000000.001'
        # If there is no main earthquake , you can set 'self.main_eq_starttime = self.time_range[0]'
        self.plot_lat = [21, 30]
        self.plot_lon = [95, 108]
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
        self.ai_catalog = 'FJ_AI2.txt'
        self.man_catalog = 'FJ_man2.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [21, 27]
        self.lon_range = [114, 122]
        self.boundary = 'province'
        self.time_range = ['20201211T000000.001', '20210206T000000.001']
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
        self.boundary = 'province'
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


class Config_XJ(object):
    def __init__(self):
        self.ai_catalog = 'XJ_AI.log'
        self.man_catalog = 'XJ_man.txt'
        self.eq_root = './data'
        self.plot_root = './image'
        self.lat_range = [35.5, 36.5]
        self.lon_range = [82.2, 83]
        # Boundary could be 'city','county','province' and 'N'
        self.boundary = 'province'
        self.time_range = ['20140212T170000.001', '20140213T180000.001']
        self.main_eq_starttime = '20140212T73609.001'
        # If there is no main earthquake , you can set 'self.main_eq_starttime = self.time_range[0]'
        self.plot_lat = [35, 37]
        self.plot_lon = [82, 83]
        self.ground_truth = 0.5
        self.cpu_cores = 120