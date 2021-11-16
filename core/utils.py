import pandas as pd
import re
from obspy import UTCDateTime, read
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def readeqman(eqf):
    with open(eqf, encoding='GBK') as f:
        lines = f.readlines()
    oklines = []
    for r in lines:
        try:
            oklines.append([r[:8], r[9:18], float(r[19:26]), float(r[27:35]), float(r[36:39]), float(r[40:45])])
        except:
            continue
    maneqpd = pd.DataFrame(oklines, columns=['ymd', 'hms', 'loclat', 'loclong', 'locdep', 'mag'])
    maneqpd['eqtime'] = pd.to_datetime(maneqpd['ymd'] + 'T' + maneqpd['hms'])
    return maneqpd


def readeqphaman(eqphaf, idline, eqf):
    # with open(eqphaf, encoding='GB18030') as f:
    with open(eqphaf, encoding='GB18030') as f:
        lines = f.readlines()
    eqpha = {}
    for line in lines[idline:]:
        sline = line.split()
        if len(sline[1]) > 8:
            eqymd = sline[1]
            eqtime = UTCDateTime(sline[1] + 'T' + sline[2])
            eqtimestr = UTCDateTime(sline[1] + 'T' + sline[2]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            key = "_".join([eqtimestr, *sline[3:6]])
            eqpha[key] = []
        else:
            if not line[0] == ' ':
                netstn = sline[0] + '.' + sline[1]
                if not line[18] == 'M':
                    pha = line[17:19]
                    ophatime = UTCDateTime(eqymd + 'T' + line[32:43])
                    if ophatime - eqtime > 0.0:
                        phatime = ophatime
                    else:
                        phatime = UTCDateTime((UTCDateTime(eqymd) + 86400).strftime('%Y-%m-%d') + 'T' + line[32:43])
                    phatimestr = phatime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                    eqpha[key].append([netstn, pha, phatimestr])
            else:
                if not line[18] == 'M':
                    pha = line[17:19]
                    ophatime = UTCDateTime(eqymd + 'T' + line[32:43])
                    if ophatime - eqtime > 0.0:
                        phatime = ophatime
                    else:
                        phatime = UTCDateTime((UTCDateTime(eqymd) + 86400).strftime('%Y-%m-%d') + 'T' + line[32:43])
                    phatimestr = phatime.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                    eqpha[key].append([netstn, pha, phatimestr])

    okeqpha = {}
    maneqpd = readeqman(eqf)
    for key in eqpha.keys():
        eqtimestr = key.split('_')[0]
        eqmagpd = maneqpd[abs((maneqpd['eqtime'] - pd.to_datetime(eqtimestr)) / np.timedelta64(1, "s")) < 1.0]
        if eqmagpd.shape[0] > 0:
            neweqtimestr = UTCDateTime(eqmagpd.ymd.values[0] + 'T' + eqmagpd.hms.values[0]).strftime(
                '%Y-%m-%dT%H:%M:%S.%f')[:-3]
            eqmag = eqmagpd.mag.values[0]
            newkey = "_".join([neweqtimestr, *key.split('_')[1:], "{:.1f}".format(eqmag)])
            okeqpha[newkey] = []
            for pha in eqpha[key]:
                okeqpha[newkey].append(pha)
    return okeqpha


def readjopenseqpha(eqphasf):
    oeqphas = {}
    with open(eqphasf, encoding='GBK') as f:
        lines = f.readlines()
    neqs = [i for i, s in enumerate(lines) if '\n' == s][0]
    for line in lines[neqs + 1:]:
        # if 'eq'==line[61:63]:
        if 'eq' in line:
            eqymd = line[3:13]
            eqtime = UTCDateTime(line[3:13] + 'T' + line[14:24]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            if not ' ' in line[26:32]:
                eqloc = [float(line[25:32]), float(line[33:41]), float(line[42:45])]
            else:
                eqloc = [float('nan'), float('nan'), float('nan')]
            if line[46:50].isspace():
                eqmag = float(line[51:54])
            else:
                eqmag = float(line[46:50])
            key = "_".join([eqtime, "{:.4f}".format(eqloc[0]), "{:.4f}".format(eqloc[1]), "{:.2f}".format(eqloc[2]),
                            "{:.1f}".format(eqmag)])
            # if key =='2021-10-30T12:03:56.600_24.1880_105.1050_7.00_1.5':
            #     breakpoint()
            oeqphas[key] = []
        else:
            if not ' ' == line[0]:
                netstn = line[0:2] + '.' + line[3:8].rstrip()
                # stn = line[3:8]
                try:
                    if line[29:30] == 'V':
                        phaname = line[17:25].rstrip()
                        if phaname in ['Pg','Pb','Pn','P']:
                            phaname='P'
                        elif phaname in ['Sg','Sb','Sn','S']:
                            phaname='S'
                        # phaname = line[17]
                        if UTCDateTime(eqymd + 'T' + line[32:43]) - UTCDateTime(eqtime) > 0.0:
                            phatime = UTCDateTime(eqymd + 'T' + line[32:43]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                        else:
                            phatime = UTCDateTime(
                                (UTCDateTime(eqymd + 'T00:00:00.000') + 86400).strftime('%Y-%m-%d') + 'T' + line[
                                                                                                            32:43]).strftime(
                                '%Y-%m-%dT%H:%M:%S.%f')[:-3]
                    else:
                        continue
                    oeqphas[key].append([netstn, phaname, phatime])
                except Exception:
                    continue
            else:
                try:
                    if line[29:30] == 'V':
                        phaname = line[17:25].rstrip()
                        # phaname = line[17]
                        if UTCDateTime(eqymd + 'T' + line[32:43]) - UTCDateTime(eqtime) > 0.0:
                            phatime = UTCDateTime(eqymd + 'T' + line[32:43]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                        else:
                            phatime = UTCDateTime(
                                (UTCDateTime(eqymd + 'T00:00:00.000') + 86400).strftime('%Y-%m-%d') + 'T' + line[
                                                                                                            32:43]).strftime(
                                '%Y-%m-%dT%H:%M:%S.%f')[:-3]
                    else:
                        continue
                    oeqphas[key].append([netstn, phaname, phatime])
                except Exception:
                    continue

    eqphas = {}
    for key in oeqphas.keys():
        if 'nan' not in key:
            eqphas[key] = oeqphas[key]

    print(len(oeqphas), len(eqphas))
    return oeqphas, eqphas


def readaieqpha(aieqphaf):
    try:
        with open(aieqphaf, encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(aieqphaf, encoding='gbk') as f:
            lines = f.readlines()
    eqphas = {}
    for line in lines:
        sline = line.split()
        if sline[0] == '#':
            eqtime = UTCDateTime(sline[2][:-4]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            if float(sline[15])>-3.0:
                eqmag=sline[15]
            else:
                eqmag="{:.2f}".format(-3.0)
            eqid = "_".join([eqtime, sline[9], sline[10], sline[11], eqmag])
            eqphas[eqid] = []
        else:
            # match = re.search("\s{7}\d\s\d\s\d\s\d\s\w", line)
            # if match is not None:
            if 'NULL' in line:
                netstn = sline[0].split('/')[0] + '.' + sline[0].split('/')[1]
                print(sline[3])
                eqphas[eqid].append(
                    [netstn, sline[3],
                     UTCDateTime(sline[4] + 'T' + sline[5]).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]])
    # with open('Associate_ALL_YnYb_0501_0601.log.isoformat', encoding='utf-8', mode='w') as f:
    #     for key in eqphas.keys():
    #         f.write('# ' + " ".join(key.split('_')) + '\n')
    #         for pha in eqphas[key]:
    #             f.write(" ".join(pha) + '\n')

    return eqphas


def callocflag(x, x0, x1, y, y0, y1):
    locflag = (x >= x0) and (x <= x1) and (y >= y0) and (y <= y1)
    return locflag


def caltimeflag(t, t0, t1):
    timeflag = ((UTCDateTime(t) - UTCDateTime(t0)) >= 0.0) and ((UTCDateTime(t1) - UTCDateTime(t)) >= 0.0)
    return timeflag


def readisoeqpha(eqphaf):
    with open(eqphaf, encoding='utf-8') as f:
        lines = f.readlines()
    eqpha = {}
    for line in lines:
        sline = line.split()
        if line == '\n':
            continue
        if sline[0] == '#':
            key = "_".join(sline[1:])
            eqpha[key] = []
        else:
            eqpha[key].append(sline)
    return eqpha


def isoeqpha2eqpd(eqphas):
    eqs = [key.split('_') for key in eqphas.keys()]
    eqpd = pd.DataFrame(eqs, columns=['timestr', 'loclat', 'loclong', 'locdep', 'mag'])
    okeqpd = eqpd.astype(dtype={'mag': 'float64', 'loclat': 'float64', 'loclong': 'float64', 'locdep': 'float64'})
    okeqpd['eqtime'] = pd.to_datetime(okeqpd['timestr'])
    return okeqpd


def wisoeqpha(eqphas, opf):
    with open(opf, encoding='utf-8', mode='w') as f:
        for key in eqphas.keys():
            f.write('# ' + " ".join(key.split('_')) + '\n')
            [f.write(" ".join(pha) + '\n') for pha in eqphas[key]]


def readstn(f):
    df = pd.read_csv(f, sep='\s+',
                     names=['net', 'stn', 'locid', 'chnhead', 'loclat', 'loclong', 'locele', 'spr'])
    stnpd = df.astype(
        dtype={'locid': 'str', 'loclat': 'float64', 'loclong': 'float64', 'locele': 'float64', 'spr': 'float64'})
    return stnpd


def eqpha2eqsetpha(eqpha):
    eqsetpha = {}
    for key in eqpha.keys():
        eqsetpha[key] = {}
        phas = eqpha[key]
        for pha in phas:
            netstn = pha[0]
            if not netstn in eqsetpha[key].keys():
                eqsetpha[key][netstn] = []
                eqsetpha[key][netstn].append(pha[1:])
            else:
                eqsetpha[key][netstn].append(pha[1:])
    return eqsetpha


def cmpeqpha(tkey, teqpha, seqpha, nmatpha_c):
    teqsetpha = eqpha2eqsetpha(teqpha)
    seqsetpha = eqpha2eqsetpha(seqpha)
    tkeyS = tkey.split('_')
    phas = teqsetpha[tkey]
    eqtime = UTCDateTime(tkeyS[0])
    nmatphastat, nmatkeys = [], []
    for key in seqsetpha.keys():
        nmatpha = 0
        if abs(UTCDateTime(key.split('_')[0]) - eqtime) <= 30.0:
            for netstn in set(phas.keys()).intersection(set(seqsetpha[key].keys())):
                for pha in phas[netstn]:
                    phaname = pha[0][0]
                    sphas = seqsetpha[key][netstn]
                    sphasphanames = [x[0][0] for x in sphas]
                    if phaname in sphasphanames:
                        inflag=0
                        for spha in sphas:
                            if phaname==spha[0][0]:
                                if abs(UTCDateTime(pha[1]) - UTCDateTime(spha[1])) < 0.5:
                                    inflag=1
                        if inflag>0: nmatpha = nmatpha + 1
        nmatphastat.append(nmatpha)
        nmatkeys.append(key)
    if max(nmatphastat) < nmatpha_c:
        return [tkey]
    else:
        return [tkey, nmatkeys[nmatphastat.index(max(nmatphastat))]]


def normalize(data):
    data -= np.mean(data, axis=0, keepdims=True)
    std_data = np.std(data, axis=0, keepdims=True)
    assert (std_data.shape[-1] == data.shape[-1])
    std_data[std_data == 0] = 1
    data /= std_data
    return data


def adjust_amplitude_for_multichannels(data):
    tmp = np.max(np.abs(data), axis=0, keepdims=True)
    assert (tmp.shape[-1] == data.shape[-1])
    if np.count_nonzero(tmp) > 0:
        data *= data.shape[-1] / np.count_nonzero(tmp)
    return data


def ploteq(key, eqphas, wfdir):
    skey = key.split('_')
    eqname = (UTCDateTime(skey[0]) - 28800).strftime('%Y%m%d%H%M%S.%f ')[:-3]
    eqmag = skey[-1].rstrip()
    phas = eqphas[key]
    netstns = set([r[0] for r in phas])
    fig, axs = plt.subplots(dpi=600, tight_layout=True)
    for netstn in netstns:
        tr = read(wfdir + '/' + eqname + '/' + netstn + '.*HZ')[0]


def readeq(eqf):
    dtype = [('ot', 'O'), ('lat', 'O'), ('lon', 'O'), ('dep', 'O'), ('mag', 'O')]
    with open(eqf) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        codes = line.split()
        ot = UTCDateTime(codes[0])
        lat, lon, dep, mag = [float(code) for code in codes[1:]]
        out.append((ot, lat, lon, dep, mag))
    return np.array(out, dtype=dtype)


def slice_ctlg(events, ot_rng=None, lat_rng=None, lon_rng=None, dep_rng=None, mag_rng=None):
    if ot_rng: events = events[(events['ot'] > ot_rng[0]) * (events['ot'] < ot_rng[1])]
    if lat_rng: events = events[(events['lat'] > lat_rng[0]) * (events['lat'] < lat_rng[1])]
    if lon_rng: events = events[(events['lon'] > lon_rng[0]) * (events['lon'] < lon_rng[1])]
    if dep_rng: events = events[(events['dep'] > dep_rng[0]) * (events['dep'] < dep_rng[1])]
    if mag_rng: events = events[(events['mag'] > mag_rng[0]) * (events['mag'] < mag_rng[1])]
    return events


def slice_ctlgv2(cfg, eqpha):
    """
    :param cfg: limited time_range and loc_range
    :param eqpha: results from readaieqpha()
    :return: same format as eqpha
    edit from /home/champak/disk2/YangbiEq/mancatlog/fan/orgnized/s_eqpha.py
    """
    okeqpha = {}
    for key in eqpha.keys():
        skey = key.split('_')
        # phas=eqpha[key]
        # npha=len(phas)
        # stns=set([pha[0] for pha in phas])
        locflag = callocflag(float(skey[1]), cfg.lat_range[0], cfg.lat_range[1], float(skey[2]), cfg.lon_range[0],
                             cfg.lon_range[1])
        timeflag = caltimeflag(skey[0], *(cfg.time_range))
        if timeflag and locflag:
            okeqpha[key] = []
            [okeqpha[key].append(r) for r in eqpha[key]]
    return okeqpha


def arrival_difference(manpha, aipha):
    """
    edit from pltphadiff.ppha
    :param manpha: manphasdf['phatime']
    :param aipha: aiphasdf['phatime']
    :return: diff
    """
    diff = (pd.to_datetime(manpha) - aipha) / np.timedelta64(1, 's')
    if diff.abs().min() <= 1.0:
        return diff.loc[diff.abs().idxmin()]


def choosebtw(v, rangelst):
    rv = (v >= rangelst[0]) and (v <= rangelst[1])
    return rv


if __name__ == '__main__':
    print()
