# Copyright (c) 2011-2013 CSIRO
# Australia Telescope National Facility (ATNF)
# Commonwealth Scientific and Industrial Research Organisation (CSIRO)
# PO Box 76, Epping NSW 1710, Australia
# atnf-enquiries@csiro.au
#
# This file is part of the ASKAP software distribution.
#
# The ASKAP software distribution is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of the License
# or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
#
"""
========================
Module :mod:`askap.time` 
========================

Utilities for manipulating and converting to and from binary atomic time (BAT).

:author: Aaron Chippendale <Aaron.Chippendale@csiro.au>
"""
# todo get DUTC0 from
# http://hpiers.obspm.fr/eoppc/bul/bulc/UTC-TAI.history

import datetime
import pytz

mjdRefDt = datetime.datetime(1858, 11, 17, tzinfo=pytz.utc)
DUTC0 = 35.0  # leap seconds - correct as of 30 June 2012

# Leap seconds definition, [UTC datetime, MJD, DUTC]
LEAPSECONDS = [
  [datetime.datetime(2015, 7, 1, tzinfo=pytz.utc), 57205, 36],
  [datetime.datetime(2012, 7, 1, tzinfo=pytz.utc), 56109, 35],
  [datetime.datetime(2009, 1, 1, tzinfo=pytz.utc), 54832, 34],
  [datetime.datetime(2006, 1, 1, tzinfo=pytz.utc), 53736, 33],
  [datetime.datetime(1999, 1, 1, tzinfo=pytz.utc), 51179, 32],
  [datetime.datetime(1997, 7, 1, tzinfo=pytz.utc), 50630, 31],
  [datetime.datetime(1996, 1, 1, tzinfo=pytz.utc), 50083, 30],
  [datetime.datetime(1994, 7, 1, tzinfo=pytz.utc), 49534, 29],
  [datetime.datetime(1993, 7, 1, tzinfo=pytz.utc), 49169, 28],
  [datetime.datetime(1992, 7, 1, tzinfo=pytz.utc), 48804, 27],
  [datetime.datetime(1991, 1, 1, tzinfo=pytz.utc), 48257, 26],
  [datetime.datetime(1990, 1, 1, tzinfo=pytz.utc), 47892, 25],
  [datetime.datetime(1988, 1, 1, tzinfo=pytz.utc), 47161, 24],
  [datetime.datetime(1985, 7, 1, tzinfo=pytz.utc), 46247, 23],
  [datetime.datetime(1993, 7, 1, tzinfo=pytz.utc), 45516, 22],
  [datetime.datetime(1982, 7, 1, tzinfo=pytz.utc), 45151, 21],
  [datetime.datetime(1981, 7, 1, tzinfo=pytz.utc), 44786, 20],
  [datetime.datetime(1980, 1, 1, tzinfo=pytz.utc), 44239, 19],
  [datetime.datetime(1979, 1, 1, tzinfo=pytz.utc), 43874, 18],
  [datetime.datetime(1978, 1, 1, tzinfo=pytz.utc), 43509, 17],
  [datetime.datetime(1977, 1, 1, tzinfo=pytz.utc), 43144, 16],
  [datetime.datetime(1976, 1, 1, tzinfo=pytz.utc), 42778, 15],
  [datetime.datetime(1975, 1, 1, tzinfo=pytz.utc), 42413, 14],
  [datetime.datetime(1974, 1, 1, tzinfo=pytz.utc), 42048, 13],
  [datetime.datetime(1973, 1, 1, tzinfo=pytz.utc), 41683, 12],
  [datetime.datetime(1972, 7, 1, tzinfo=pytz.utc), 41499, 11],
  [datetime.datetime(1972, 1, 1, tzinfo=pytz.utc), 41317, 10],
  ]


class SiteError(Exception):
    pass


def getDUTCDt(dt):
    """
    Get the DUTC value in seconds that applied at the given (datetime)
    timestamp.
    """
    for i in LEAPSECONDS:
        if i[0] < dt:
            return i[2]
    # Different system used before then
    return 0


def getDUTCMJD(dt):
    """
    Get the DUTC value in seconds that applied at the given (MJD) timestamp.
    """
    for i in LEAPSECONDS:
        if i[1] < dt:
            return i[2]
    # Different system used before then
    return 0
    

def bat2utc(bat, dutc=DUTC0):
    """
    Convert Binary Atomic Time (BAT) to UTC.  At the ATNF, BAT corresponds to
    the number of microseconds of atomic clock since MJD (1858-11-17 00:00).

    :param bat: number of microseconds of atomic clock time since
        MJD 1858-11-17 00:00.
    :type bat: long
    :returns utcDJD: UTC date and time (Dublin Julian Day as used by pyephem)
    :rtype: float

    """
    utcMJDs = (bat/1000000.0)-dutc
    utcDJDs = utcMJDs-86400.0*15019.5
    utcDJD = utcDJDs/86400.0
    return utcDJD


def bat_now():
    """Return current time as  (long value) BAT"""    
    return int(utcDt2batStr(datetime.datetime.now(pytz.utc)), 16)


def utc2bat(utc, dutc=DUTC0):
    """
    Convert UTC to Binary Atomic Time (BAT).  At the ATNF, BAT corresponds to
    the number of microseconds of atomic clock since MJD (1858-11-17 00:00).

    :param utc: Universal coordinated time as a Dublin Julian Day.
    :type utc: float
    :returns bat: Binary Atomic Time (BAT) = number of microseconds of atomic
                clock time since MJD 1858-11-17 00:00.
    :rtype: long

    """
    # utcDJDs = 86400.0*utc
    utcMJDs = 86400.0*(utc+15019.5)  # in seconds
    bat = int(utcMJDs + dutc)*1000000
    return bat


def utcdjd2bat(utcDJD, dutc=DUTC0):
    utcDJDs = utcDJD*86400.0
    utcMJDs = utcDJDs + 86400.0*15019.5
    bat = int(utcMJDs + dutc)*1000000
    return bat


def bat2utcDt(bat, dutc=DUTC0):
    """
    Convert Binary Atomic Time (BAT) to UTC as a datetime object.

    :author: Douglas Hayman <douglas.hayman@csiro.au>

    :param bat: int or long (microseconds) or string (hex, microseconds)
    :rtype: :class:`datetime.datetime`

    """
    if type(bat) is str:
        bat = int(bat, base=16)
    # elif not(isinstance(bat, numbers.Number)):
    # elif not((type(bat) is int) or (type(bat) is long)):
    #    print('Argument bat must be a string or a number, not %s'% type(bat))
    #    raise(TypeError)

    # Add MJD date and subtract the TIA correction
    batDt = datetime.timedelta(microseconds=bat - dutc*1.0e6) + mjdRefDt
    return batDt


def utcDt2bat(utcDt, dutc=DUTC0):
    """
    Convert UTC as a datetime object to BAT number

    :author: Keith Bannister

    :param datetime.datetime utcDt:
    :param float dutc: leap second correction
    :returns: **bat**  microseconds 
    :rtype: *int*
    """
    dt = utcDt - mjdRefDt  # generates a datetime.deltatime object
    bat = dt.microseconds + (dt.seconds + int(dutc) + dt.days*24*3600)*1000000

    return bat


def utcDt2batStr(utcDt, dutc=DUTC0):
    """
    Convert UTC as a datetime object to BAT hex string for calling MoniCA

    :author: Douglas Hayman <douglas.hayman@csiro.au>

    :param datetime.datetime utcDt:
    :param float dutc: leap second correction
    :returns: **bat**  microseconds in hex eg. 0x11465c69b38388
    :rtype: *string*
    """
    dt = utcDt - mjdRefDt  # generates a datetime.deltatime object
    bat = dt.microseconds + (dt.seconds + dutc + dt.days*24*3600)*10**6
    # python casts this into an int. It would cast it into long if needed
    return '0x%x' % bat


def siteID2LocTimezone(siteID):
    """
    Convert the siteID to a datetime timezone object

    :param str siteID: PKS, MRO or MATES
    :rtype: datetime.tzinfo

    """
    if siteID in ['PKS', 'MATES']:
        tzStr = 'Australia/Sydney'
    elif siteID == 'MRO':
        tzStr = 'Australia/Perth'
    else:
        errorMsg = 'Unknown siteID: %s\n\n' % siteID
        raise SiteError(errorMsg)
    return pytz.timezone(tzStr)


def isoUTC2datetime(iso):
    """Convert and ISO8601 (UTC only) like string date/time value to a
    :obj:`datetime.datetime` object.

    :param str iso: ISO8061 string
    :rtype: datetime.datetime
    """
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]
    if 'T' in iso:
        formats = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                   "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(iso, fmt)
        except ValueError:
            continue
    raise ValueError("Couldn't parse ISO8061 string '{}'".format(iso))

def utc_now():
    """
    Return current UTC as a timezone aware datetime.
    :rtype: datetime
    """
    dt=datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    return dt

def utcDt2mjd(utcDt):
    """
    Convert UTC as a datetime object to Modified Julian Date.

    :param datetime.datetime utcDt:
    :rtype: float
    """
    dt = utcDt - mjdRefDt  # generates a datetime.deltatime object
    mjd = dt.days + dt.seconds/86400.0 + dt.microseconds/86400000000.0
    return mjd
