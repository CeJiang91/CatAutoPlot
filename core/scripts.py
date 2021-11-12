from core.config import Config as Config
from core.utils import readisoeqpha, cmpeqpha, wisoeqpha, readaieqpha, readjopenseqpha, \
    slice_ctlgv2, readeq, arrival_difference
import os
import multiprocessing
from functools import partial
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from obspy.geodetics import locations2degrees, degrees2kilometers
from obspy import UTCDateTime
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
import math

mpl.use('TkAgg')


def log2eqpha(cfg=Config()):
    eq_root = cfg.eq_root
    ai_catalog = cfg.ai_catalog
    man_catalog = cfg.man_catalog
    if not os.path.isdir(eq_root):
        os.makedirs(eq_root)
    aieqpha = readaieqpha(os.path.join(eq_root, ai_catalog))
    aieqpha = slice_ctlgv2(cfg, aieqpha)  # fan
    wisoeqpha(aieqpha, os.path.join(eq_root, 'aieqpha.dat'))
    _, maneqpha = readjopenseqpha(os.path.join(eq_root, man_catalog))
    maneqpha = slice_ctlgv2(cfg, maneqpha)  # fan
    wisoeqpha(maneqpha, os.path.join(eq_root, 'maneqpha.dat'))
    with open(os.path.join(eq_root, 'aieqs.lst'), mode='w') as f:
        [f.write(" ".join(r.split('_')) + '\n') for r in aieqpha.keys()]
    print("number of ai_earthquakes: ", len(aieqpha.keys()))
    with open(os.path.join(eq_root, 'maneqs.lst'), mode='w') as f:
        [f.write(" ".join(r.split('_')) + '\n') for r in maneqpha.keys()]
    print("number of man_earthquakes: ", len(maneqpha.keys()))


def data_cmp(cfg=Config()):
    print('data_cmp')
    aieqpha = readisoeqpha(os.path.join(cfg.eq_root, 'aieqpha.dat'))
    maneqpha = readisoeqpha(os.path.join(cfg.eq_root, 'maneqpha.dat'))
    cmppath = os.path.join(cfg.eq_root, 'cmp_results')
    if os.path.isdir(cmppath):
        shutil.rmtree(cmppath)
    os.makedirs(cmppath)
    p = multiprocessing.Pool(cfg.cpu_cores)
    manresults = p.map(partial(cmpeqpha, teqpha=maneqpha, seqpha=aieqpha, nmatpha_c=2), list(maneqpha.keys()))
    p.close()
    p.join()
    hmannotcat = open(os.path.join(cmppath, 'man_nocat.dat'), mode='w', encoding='utf-8')
    hmancatman = open(os.path.join(cmppath, 'man_catman.dat'), mode='w', encoding='utf-8')
    hmancatai = open(os.path.join(cmppath, 'man_catai.dat'), mode='w', encoding='utf-8')
    hmannotcateq = open(os.path.join(cmppath, 'man_nocateq.dat'), mode='w', encoding='utf-8')
    hmancatmaneq = open(os.path.join(cmppath, 'man_catmaneq.dat'), mode='w', encoding='utf-8')
    hmancataieq = open(os.path.join(cmppath, 'man_cataieq.dat'), mode='w', encoding='utf-8')
    i, j = 1, 1
    for r in manresults:
        if len(r) == 1:
            hmannotcat.write('# ' + " ".join(r[0].split('_')) + '\n')
            hmannotcateq.write('# ' + " ".join(r[0].split('_')) + '\n')
            [hmannotcat.write(" ".join(pha) + '\n') for pha in maneqpha[r[0]]]
            i = i + 1
        else:
            hmancatman.write('# ' + " ".join(r[0].split('_')) + '\n')
            hmancatmaneq.write('# ' + " ".join(r[0].split('_')) + '\n')
            [hmancatman.write(" ".join(pha) + '\n') for pha in maneqpha[r[0]]]
            hmancatai.write('# ' + " ".join(r[1].split('_')) + '\n')
            hmancataieq.write('# ' + " ".join(r[1].split('_')) + '\n')
            [hmancatai.write(" ".join(pha) + '\n') for pha in aieqpha[r[1]]]
            j = j + 1
    hmannotcat.close()
    hmancatman.close()
    hmancatai.close()
    hmannotcateq.close()
    hmancatmaneq.close()
    hmancataieq.close()

    p = multiprocessing.Pool(cfg.cpu_cores)
    airesults = p.map(partial(cmpeqpha, teqpha=aieqpha, seqpha=maneqpha, nmatpha_c=3), list(aieqpha.keys()))
    p.close()
    p.join()
    hainotcat = open(os.path.join(cmppath, 'ai_nocat.dat'), mode='w', encoding='utf-8')
    haicatai = open(os.path.join(cmppath, 'ai_catai.dat'), mode='w', encoding='utf-8')
    haicatman = open(os.path.join(cmppath, 'ai_catman.dat'), mode='w', encoding='utf-8')
    hainotcateq = open(os.path.join(cmppath, 'ai_nocateq.dat'), mode='w', encoding='utf-8')
    haicataieq = open(os.path.join(cmppath, 'ai_cataieq.dat'), mode='w', encoding='utf-8')
    haicatmaneq = open(os.path.join(cmppath, 'ai_catmaneq.dat'), mode='w', encoding='utf-8')
    i, j = 1, 1
    for r in airesults:
        if len(r) == 1:
            hainotcat.write('# ' + " ".join(r[0].split('_')) + '\n')
            hainotcateq.write('# ' + " ".join(r[0].split('_')) + '\n')
            [hainotcat.write(" ".join(pha) + '\n') for pha in aieqpha[r[0]]]
            i = i + 1
        else:
            haicatai.write('# ' + " ".join(r[0].split('_')) + '\n')
            haicataieq.write('# ' + " ".join(r[0].split('_')) + '\n')
            [haicatai.write(" ".join(pha) + '\n') for pha in aieqpha[r[0]]]
            haicatman.write('# ' + " ".join(r[1].split('_')) + '\n')
            haicatmaneq.write('# ' + " ".join(r[1].split('_')) + '\n')
            [haicatman.write(" ".join(pha) + '\n') for pha in maneqpha[r[1]]]
            j = j + 1
    hainotcat.close()
    haicatai.close()
    haicatman.close()
    hainotcateq.close()
    haicataieq.close()
    haicatmaneq.close()


def location_cmp(cfg=Config()):
    print('location_cmp')
    fig = plt.figure(dpi=600)
    plt.rc('font', family='Nimbus Roman')
    ax = plt.axes(projection=ccrs.PlateCarree())
    proj = ccrs.PlateCarree()
    ax.set_extent([cfg.plot_lon[0], cfg.plot_lon[1], cfg.plot_lat[0], cfg.plot_lat[1]], ccrs.Geodetic())
    boundary = cfg.boundary
    if boundary != 'N':
        states_shp = os.path.join(cfg.eq_root, f'shp/{boundary}')
        for state in shpreader.Reader(states_shp).geometries():
            edgecolor = 'black'
            ax.add_geometries([state], ccrs.PlateCarree(), facecolor='none', edgecolor=edgecolor)
    ax.imshow(
        imread(os.path.join(cfg.eq_root, 'NE1_50M_SR_W.tif')),
        origin='upper', transform=proj,
        extent=[-180, 180, -90, 90]
    )
    maneqs = readeq(os.path.join(cfg.eq_root, 'maneqs.lst'))
    aieqs = readeq(os.path.join(cfg.eq_root, 'aieqs.lst'))
    manlat = maneqs['lat'].astype(np.float)
    manlon = maneqs['lon'].astype(np.float)
    ailat = aieqs['lat'].astype(np.float)
    ailon = aieqs['lon'].astype(np.float)
    plt.scatter(ailon, ailat, c='navy', marker='o', s=0.5)
    plt.scatter(manlon, manlat, c='r', marker='o', s=0.5)
    frame = plt.gca()  #
    frame.axes.get_yaxis().set_visible(True)
    frame.axes.get_xaxis().set_visible(True)
    plt.xlabel('Longitude', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    plt.ylabel('Latitude', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    if not os.path.isdir(cfg.plot_root):
        os.makedirs(cfg.plot_root)
    fig.savefig(os.path.join(cfg.plot_root, 'loc_lon_lat.png'), format='png')
    plt.close()
    mandepth = maneqs['dep'].astype(np.float)
    aidepth = aieqs['dep'].astype(np.float)
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.rc('font', family='Nimbus Roman')
    frame = plt.gca()
    frame.invert_yaxis()
    plt.scatter(ailon, aidepth, c='navy', marker='o', s=0.5)
    plt.scatter(manlon, mandepth, c='r', marker='o', s=0.5)
    plt.ylim([30, 0])
    plt.xlabel('Longitude', fontdict={'family': 'Nimbus Roman',
                                      'weight': 'normal', 'size': 12})
    plt.ylabel('Depth', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 12})
    fig.savefig(os.path.join(cfg.plot_root, 'loc_lon_dep.png'), format='png', bbox_inches='tight')
    plt.close()
    fig = plt.figure(figsize=(5, 3), dpi=200)
    plt.rc('font', family='Nimbus Roman')
    frame = plt.gca()
    frame.invert_yaxis()
    plt.scatter(ailat, aidepth, c='navy', marker='o', s=0.5)
    plt.scatter(manlat, mandepth, c='r', marker='o', s=0.5)
    plt.ylim([30, 0])
    plt.xlabel('Latitude', fontdict={'family': 'Nimbus Roman',
                                     'weight': 'normal', 'size': 12})
    plt.ylabel('Depth', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 12})
    fig.savefig(os.path.join(cfg.plot_root, 'loc_lat_dep.png'), format='png', bbox_inches='tight')
    plt.close()


def factor_cmp(cfg=Config()):
    print('factor_cmp')
    font_manager._rebuild()
    aieqs = pd.read_csv(os.path.join(cfg.eq_root, 'cmp_results/man_cataieq.dat'), sep='\s+', header=None,
                        names=['tmp', 'timestr', 'loclat', 'loclong', 'locdep', 'mag'],
                        dtype={'loclat': 'float64', 'loclong': 'float64', 'locdep': 'float64', 'mag': 'float64'})
    aieqs['eqtime'] = pd.to_datetime(aieqs['timestr'])

    maneqs = pd.read_csv(os.path.join(cfg.eq_root, 'cmp_results/man_catmaneq.dat'), sep='\s+', header=None,
                         names=['tmp', 'timestr', 'loclat', 'loclong', 'locdep', 'mag'],
                         dtype={'loclat': 'float64', 'loclong': 'float64', 'locdep': 'float64', 'mag': 'float64'})
    maneqs['eqtime'] = pd.to_datetime(maneqs['timestr'])
    # paras={'font.family':'Times New Roman'}
    paras = {'font.family': 'Nimbus Roman'}
    plt.rcParams.update(paras)
    fig, axs = plt.subplots(2, 2, tight_layout=True, dpi=600, figsize=[7, 5])
    # plt.rc('font', family='Nimbus Roman')
    axs[0, 0].hist((aieqs.eqtime - maneqs.eqtime) / np.timedelta64(1, 's'), edgecolor='black',
                   bins=np.arange(-4.0, 4.0, 0.2))
    axs[0, 0].set_title('Origin time differences')
    axs[0, 0].set_xlabel(r'$OT_{AI}-OT_{Man}(s)$')
    axs[0, 0].set_xlim(-5, 5)
    dists = []
    for i in range(aieqs.shape[0]):
        ailat, ailong = aieqs.loc[i].loclat, aieqs.loc[i].loclong
        manlat, manlong = maneqs.loc[i].loclat, maneqs.loc[i].loclong
        dists.append(degrees2kilometers(locations2degrees(ailat, ailong, manlat, manlong)))
    axs[0, 1].hist(dists, edgecolor='black', bins=np.arange(0.0, max(dists), 1))
    axs[0, 1].set_title('Distance between epicenters')
    axs[0, 1].set_xlabel('$Distance (km)$')
    axs[0, 1].set_xlim(right=max(dists) + 0.25)
    axs[1, 0].hist(aieqs.locdep - maneqs.locdep, edgecolor='black',
                   bins=np.arange((aieqs.locdep - maneqs.locdep).min() - 1.0,
                                  (aieqs.locdep - maneqs.locdep).max() + 1.0, 1.0))
    axs[1, 0].set_title(r'Depth differences')
    axs[1, 0].set_xlabel(r'$Dep_{AI}-Dep_{Man}(km)$')
    axs[1, 0].set_xlim((aieqs.locdep - maneqs.locdep).min() - 1.0, (aieqs.locdep - maneqs.locdep).max() + 1.0)
    axs[1, 1].hist(aieqs.mag[aieqs.mag > -10.0] - maneqs.mag[aieqs.mag > -10.0], edgecolor='black',
                   bins=np.arange(-1.0, 1.0, 0.1))
    axs[1, 1].set_title('Magnitude differences')
    axs[1, 1].set_xlabel(r'$M_{L}\_AI-M_{L}\_Man$')
    axs[1, 1].set_xlim(-1, 1)
    plt.savefig(os.path.join(cfg.plot_root, 'diff.png'), format='png')
    plt.close()


def pha_cmp(cfg=Config()):
    print('pha_cmp')
    aiphas = []
    with open(os.path.join(cfg.eq_root, 'aieqpha.dat')) as f:
        lines = f.readlines()
    for line in lines:
        if not line[0] == '#':
            aiphas.append(line.split())
    aiphasdf = pd.DataFrame(aiphas, columns=['netstn', 'phaname', 'timestr'])
    aiphasdf['phatime'] = pd.to_datetime(aiphasdf['timestr'])
    manphas = []
    with open(os.path.join(cfg.eq_root, 'maneqpha.dat')) as f:
        lines = f.readlines()
    for line in lines:
        if not line[0] == '#':
            manphas.append(line.split())
    manphasdf = pd.DataFrame(manphas, columns=['netstn', 'phaname', 'timestr'])
    manphasdf['phatime'] = pd.to_datetime(manphasdf['timestr'])
    p = multiprocessing.Pool(cfg.cpu_cores)
    results_Pg = p.map(partial(arrival_difference, aipha=aiphasdf[aiphasdf['phaname'].str.contains('P')]['phatime']),
                       manphasdf[manphasdf['phaname'].str.contains('P')]['phatime'])
    difftimes_Pg = []
    for rr in list(filter(None.__ne__, results_Pg)):
        difftimes_Pg.append(rr)
    results_Sg = p.map(partial(arrival_difference, aipha=aiphasdf[aiphasdf['phaname'].str.contains('S')]['phatime']),
                       manphasdf[manphasdf['phaname'].str.contains('S')]['phatime'])
    difftimes_Sg = []
    for rr in list(filter(None.__ne__, results_Sg)):
        difftimes_Sg.append(rr)
    p.close()
    p.join()
    plt.figure(tight_layout=True, dpi=600)
    plt.rc('font', family='Nimbus Roman')
    plt.hist(difftimes_Pg, bins=np.arange(-1.0, 1.0, 0.02), weights=np.ones(len(difftimes_Pg)) / len(difftimes_Pg),
             edgecolor='red', linewidth=1, facecolor='red', range=range, alpha=0.3, label='Pg')
    plt.hist(difftimes_Sg, bins=np.arange(-1.0, 1.0, 0.02), weights=np.ones(len(difftimes_Sg)) / len(difftimes_Sg),
             edgecolor='blue', linewidth=1, facecolor='blue', range=range, alpha=0.3, label='Sg')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(2))
    plt.xlabel('${T_{AI}}$ - ${T_{Catalog}}$', fontdict={'family': 'Nimbus Roman',
                                                         'weight': 'normal', 'size': 15})
    plt.ylabel('Percentage', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(os.path.join(cfg.plot_root, 'phatimediff.png'), format='png')


def indicator_cmp(cfg=Config()):
    ai_all = readisoeqpha(os.path.join(cfg.eq_root, 'aieqpha.dat'))
    aicat = readisoeqpha(os.path.join(cfg.eq_root, 'cmp_results', 'man_catai.dat'))
    man_all = readisoeqpha(os.path.join(cfg.eq_root, 'maneqpha.dat'))
    eq_recall = aicat.__len__() / man_all.__len__()
    eq_precision = aicat.__len__() / ai_all.__len__()
    eq_f1 = 2 * (eq_precision * eq_recall) / (eq_recall + eq_precision)
    # -------------------
    aicatphas = []
    with open(os.path.join(cfg.eq_root, 'cmp_results', 'man_catai.dat')) as f:
        lines = f.readlines()
    for line in lines:
        if not line[0] == '#':
            aicatphas.append(line.split())
    aicatphasdf = pd.DataFrame(aicatphas, columns=['netstn', 'phaname', 'timestr'])
    aicatphasdf['phatime'] = pd.to_datetime(aicatphasdf['timestr'])
    aiallphas = []
    with open(os.path.join(cfg.eq_root, 'aieqpha.dat')) as f:
        lines = f.readlines()
    for line in lines:
        if not line[0] == '#':
            aiallphas.append(line.split())
    aiallphasdf = pd.DataFrame(aiallphas, columns=['netstn', 'phaname', 'timestr'])
    aiallphasdf['phatime'] = pd.to_datetime(aiallphasdf['timestr'])
    manphas = []
    with open(os.path.join(cfg.eq_root, 'maneqpha.dat')) as f:
        lines = f.readlines()
    for line in lines:
        if not line[0] == '#':
            manphas.append(line.split())
    manphasdf = pd.DataFrame(manphas, columns=['netstn', 'phaname', 'timestr'])
    manphasdf['phatime'] = pd.to_datetime(manphasdf['timestr'])
    p = multiprocessing.Pool(cfg.cpu_cores)
    results_Pg = p.map(partial(arrival_difference, aipha=aicatphasdf[aicatphasdf.phaname == 'Pg']['phatime']),
                       manphasdf[manphasdf.phaname == 'Pg']['phatime'])
    difftimes_Pg = []
    for rr in list(filter(None.__ne__, results_Pg)):
        difftimes_Pg.append(rr)
    results_Sg = p.map(partial(arrival_difference, aipha=aicatphasdf[aicatphasdf.phaname == 'Sg']['phatime']),
                       manphasdf[manphasdf.phaname == 'Sg']['phatime'])
    difftimes_Sg = []
    for rr in list(filter(None.__ne__, results_Sg)):
        difftimes_Sg.append(rr)
    p.close()
    p.join()
    ground_truth = cfg.ground_truth
    difftimes_Pg = np.array(difftimes_Pg)
    Pg_diff = difftimes_Pg[difftimes_Pg <= ground_truth]
    Pg_recall = Pg_diff.shape[0] / manphasdf[manphasdf.phaname == 'Pg']['phatime'].__len__()
    Pg_precision = Pg_diff.shape[0] / aiallphasdf[aiallphasdf.phaname == 'Pg'].shape[0]
    Pg_f1 = 2 * (Pg_precision * Pg_recall) / (Pg_precision + Pg_recall)
    difftimes_Sg = np.array(difftimes_Sg)
    Sg_diff = difftimes_Sg[difftimes_Sg <= ground_truth]
    Sg_recall = Sg_diff.shape[0] / manphasdf[manphasdf.phaname == 'Sg']['phatime'].__len__()
    Sg_precision = Sg_diff.shape[0] / aiallphasdf[aiallphasdf.phaname == 'Sg'].shape[0]
    Sg_f1 = 2 * (Sg_precision * Sg_recall) / (Sg_precision + Sg_recall)
    with open(os.path.join(cfg.plot_root, 'indicator_report.txt'), 'w') as the_file:
        the_file.write(f"Earthquake-------------------------------\n")
        the_file.write(f"recall = {'%5.3f' % eq_recall}, precision = {'%5.3f' % eq_precision}, "
                       f"f1_score = {'%5.3f' % eq_f1} \n")
        the_file.write(f"man earthquake = {'%5d' % man_all.__len__()}, ai earthquake = {'%5d' % ai_all.__len__()}, "
                       f"TP = {'%5d' % aicat.__len__()} \n")
        the_file.write(f'Phase------------------------------------\n')
        the_file.write(f"Pg: recall = {'%5.3f' % Pg_recall}, precision = {'%5.3f' % Pg_precision}, "
                       f"f1_score = {'%5.3f' % Pg_f1}\n")
        the_file.write(f"Pg: std = {'%5.3f' % np.std(Pg_diff)} var = {'%5.3f' % np.var(Pg_diff)}, "
                       f"mean/abs = {'%5.3f' % np.mean(np.abs(Pg_diff))} mean = {'%5.3f' % np.mean(Pg_diff)}\n")
        the_file.write(f"Sg: recall = {'%5.3f' % Sg_recall}, precision = {'%5.3f' % Sg_precision},"
                       f" f1_score = {'%5.3f' % Sg_f1}\n")
        the_file.write(f"Sg: std = {'%5.3f' % np.std(Sg_diff)} var = {'%5.3f' % np.var(Sg_diff)}, "
                       f"mean/abs = {'%5.3f' % np.mean(np.abs(Sg_diff))} mean = {'%5.3f' % np.mean(Sg_diff)}\n")


def mag_cmp(cfg=Config()):
    print('mag_cmp')
    aieqs = pd.read_csv(os.path.join(cfg.eq_root, 'aieqs.lst'), sep='\s+', header=None,
                        names=['timestr', 'loclat', 'loclong', 'locdep', 'mag'],
                        dtype={'loclat': 'float64', 'loclong': 'float64', 'locdep': 'float64', 'mag': 'float64'})
    aieqs['eqtime'] = pd.to_datetime(aieqs['timestr'])

    maneqs = pd.read_csv(os.path.join(cfg.eq_root, 'maneqs.lst'), sep='\s+', header=None,
                         names=['timestr', 'loclat', 'loclong', 'locdep', 'mag'],
                         dtype={'loclat': 'float64', 'loclong': 'float64', 'locdep': 'float64', 'mag': 'float64'})
    maneqs['eqtime'] = pd.to_datetime(maneqs['timestr'])
    plt.figure(tight_layout=True, figsize=(6, 4), dpi=200)
    plt.rc('font', family='Nimbus Roman')
    plt.hist(aieqs.mag, bins=np.arange(-1.0, 5.0, 0.1),
             edgecolor='red', linewidth=1, facecolor='red', range=range, alpha=0.3, label='AI')
    plt.hist(maneqs.mag, bins=np.arange(-1.0, 5.0, 0.1),
             edgecolor='blue', linewidth=1, facecolor='blue', range=range, alpha=0.3, label='Man')
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(2))
    plt.xlabel('Magnitude($\mathregular{M_{L}}$)', fontdict={'family': 'Nimbus Roman',
                                                             'weight': 'normal', 'size': 15})
    plt.ylabel('Counts', fontdict={'family': 'Nimbus Roman', 'weight': 'normal', 'size': 15})
    plt.legend(loc="best")
    plt.savefig(os.path.join(cfg.plot_root, 'Mag_hist.png'), format='png')
    plt.close()
    # ----------------


def plot_MT(cfg=Config()):
    """
    edit from fan:pltmt.py
    :param cfg:
    :return:
    """
    print('plot MT')
    maineqtime = cfg.main_eq_starttime
    with open(os.path.join(cfg.eq_root, 'maneqs.lst')) as f:
        lines = f.readlines()
    oklines = []
    for r in lines:
        sr = r.split()
        oklines.append([sr[0], float(sr[1]), float(sr[2]), float(sr[3]), float(sr[4])])
    omaneqpd = pd.DataFrame(oklines, columns=['eqtimestr', 'loclat', 'loclong', 'locdep', 'mag'])
    omaneqpd['eqtime'] = pd.to_datetime(omaneqpd['eqtimestr'])
    omaneqpd['dt'] = (omaneqpd['eqtime'] - pd.to_datetime(maineqtime)) / np.timedelta64(1, "s")
    maneq = omaneqpd
    omags = np.array(maneq.mag)
    otimes = np.array(maneq.dt)
    manmags = omags[omags >= -3.0]
    mantimes = otimes[omags >= -3.0]

    with open(os.path.join(cfg.eq_root, 'aieqs.lst')) as f:
        lines = f.readlines()
    oklines = []
    for r in lines:
        sr = r.split()
        oklines.append([sr[0], float(sr[1]), float(sr[2]), float(sr[3]), float(sr[4])])
    oaieqpd = pd.DataFrame(oklines, columns=['eqtimestr', 'loclat', 'loclong', 'locdep', 'mag'])
    oaieqpd['eqtime'] = pd.to_datetime(oaieqpd['eqtimestr'])
    oaieqpd['dt'] = (oaieqpd['eqtime'] - pd.to_datetime(maineqtime)) / np.timedelta64(1, "s")
    aieq = oaieqpd
    omags = np.array(aieq.mag - 0.17)
    otimes = np.array(aieq.dt)
    aimags = omags[omags >= -3.0]
    aitimes = otimes[omags >= -3.0]

    config = cfg
    otimedt_day = math.floor(
        ((pd.to_datetime(config.time_range[0]) - pd.to_datetime(maineqtime)) / np.timedelta64(1, 's')) / 86400)
    etimedt_day = math.floor(
        ((pd.to_datetime(config.time_range[1]) - pd.to_datetime(maineqtime)) / np.timedelta64(1, 's')) / 86400) + 1
    otimedt_hour = math.floor(
        ((pd.to_datetime(config.time_range[0]) - pd.to_datetime(maineqtime)) / np.timedelta64(1, 's')) / 3600)
    etimedt_hour = math.floor(
        ((pd.to_datetime(config.time_range[1]) - pd.to_datetime(maineqtime)) / np.timedelta64(1, 's')) / 3600) + 1

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 5), dpi=200)
    fig.subplots_adjust(hspace=0)
    axs[0].scatter(aitimes, aimags, s=3.0, marker='.', label='AI')
    axs[0].scatter(mantimes, manmags, s=3.0, marker='.', label='man')
    axs[0].axvline(color='red', linewidth=1.0)
    axs[0].plot(0.0, np.max(manmags), 'r*', fillstyle='none', markeredgewidth=1.0)
    axs[0].set_ylabel('Magnitude')
    axs[0].set_xlim(otimedt_day * 86400, etimedt_day * 86400)
    axs[0].legend(loc='upper left')

    saimags = aimags[np.argsort(aitimes)]
    saitimes = aitimes[np.argsort(aitimes)]
    aivalues, aibase = np.histogram(saitimes, bins=saitimes)
    aicum = np.cumsum(aivalues)
    axs[1].plot(saitimes[1:], aicum, '--', label='AI')
    axs[1].axvline(color='red', linewidth=1.0)
    smanmags = manmags[np.argsort(mantimes)]
    smantimes = mantimes[np.argsort(mantimes)]
    manvalues, manbase = np.histogram(smantimes, bins=smantimes)
    mancum = np.cumsum(manvalues)
    axs[1].plot(smantimes[1:], mancum, '-', label='man')
    axs[1].axvline(color='red', linewidth=1.0)
    axs[1].set_ylabel('Cumulative number')
    axs[1].legend(loc='upper left')
    xlabels = ["{:} days".format(r) for r in range(otimedt_day, 0, 1)] + \
              [UTCDateTime(maineqtime).strftime('%m-%d/%H:%M')] + \
              ["{:} days".format(r) for r in range(1, etimedt_day, 1)]
    plt.xticks([r * 86400 for r in range(otimedt_day, etimedt_day, 1)], xlabels, rotation=45)
    axs[1].set_xticks([r * 3600 for r in range(math.floor(otimedt_hour), math.floor(etimedt_hour), 2)], minor=True)
    axs[0].set_xticks([r * 3600 for r in range(math.floor(otimedt_hour), math.floor(etimedt_hour), 2)], minor=True)
    axs[0].tick_params(axis='x', which='major', color='red')
    axs[1].tick_params(axis='x', which='major', color='red')
    axs[1].xaxis.grid(True, which='major')
    axs[1].yaxis.grid(True, which='major')
    axs[0].xaxis.grid(True, which='major')
    plt.savefig(os.path.join(cfg.plot_root, 'mt.png'), format='png', bbox_inches='tight')
    plt.close()


def traveltime_cmp(config=Config()):
    print('traveltime_cmp')


def eqdensity_cmp(config=Config()):
    print('eqdensity_cmp')
