"""
Copyright (C) 2019 Interactive Brokers LLC. All rights reserved. This code is subject to the terms
 and conditions of the IB API Non-Commercial License or the IB API Commercial License, as applicable.
"""

import sys
import ibapi
import math

from decimal import Decimal
from ibapi.enum_implem import Enum
from ibapi.object_implem import Object


NO_VALID_ID = -1
MAX_MSG_LEN = 0xFFFFFF # 16Mb - 1byte

UNSET_INTEGER = 2 ** 31 - 1
UNSET_DOUBLE = sys.float_info.max
UNSET_LONG = 2 ** 63 - 1
UNSET_DECIMAL = Decimal(2 ** 127 - 1)
DOUBLE_INFINITY = math.inf

INFINITY_STR = "Infinity"

TickerId = int
OrderId  = int
TagValueList = list

FaDataType = int
FaDataTypeEnum = Enum("N/A", "GROUPS", "PROFILES", "ALIASES")

MarketDataType = int
MarketDataTypeEnum = Enum("N/A", "REALTIME", "FROZEN", "DELAYED", "DELAYED_FROZEN")

Liquidities = int
LiquiditiesEnum = Enum("None", "Added", "Remove", "RoudedOut")

SetOfString = set
SetOfFloat = set
ListOfOrder = list
ListOfFamilyCode = list
ListOfContractDescription = list
ListOfDepthExchanges = list
ListOfNewsProviders = list
SmartComponentMap = dict
HistogramDataList = list
ListOfPriceIncrements = list
ListOfHistoricalTick = list
ListOfHistoricalTickBidAsk = list
ListOfHistoricalTickLast = list
ListOfHistoricalSessions = list


class BarData(Object):
    def __init__(self):
        self.date = ""
        self.open = 0.
        self.high = 0.
        self.low = 0.
        self.close = 0.
        self.volume = UNSET_DECIMAL
        self.wap = UNSET_DECIMAL
        self.barCount = 0

    def __str__(self):
        return "Date: %s, Open: %s, High: %s, Low: %s, Close: %s, Volume: %s, WAP: %s, BarCount: %s" % (self.date, ibapi.utils.floatMaxString(self.open), 
            ibapi.utils.floatMaxString(self.high), ibapi.utils.floatMaxString(self.low), ibapi.utils.floatMaxString(self.close), 
            ibapi.utils.decimalMaxString(self.volume), ibapi.utils.decimalMaxString(self.wap), ibapi.utils.intMaxString(self.barCount))


class RealTimeBar(Object):
    def __init__(self, time = 0, endTime = -1, open_ = 0., high = 0., low = 0., close = 0., volume = UNSET_DECIMAL, wap = UNSET_DECIMAL, count = 0):
        self.time = time
        self.endTime = endTime
        self.open_ = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.wap = wap
        self.count = count

    def __str__(self):
        return "Time: %s, Open: %s, High: %s, Low: %s, Close: %s, Volume: %s, WAP: %s, Count: %s" % (ibapi.utils.intMaxString(self.time), 
            ibapi.utils.floatMaxString(self.open), ibapi.utils.floatMaxString(self.high), ibapi.utils.floatMaxString(self.low), 
            ibapi.utils.floatMaxString(self.close), ibapi.utils.decimalMaxString(self.volume), ibapi.utils.decimalMaxString(self.wap), 
            ibapi.utils.intMaxString(self.count))


class HistogramData(Object):
    def __init__(self):
        self.price = 0.
        self.size = UNSET_DECIMAL

    def __str__(self):
        return "Price: %s, Size: %s" % (ibapi.utils.floatMaxString(self.price), ibapi.utils.decimalMaxString(self.size))


class NewsProvider(Object):
    def __init__(self):
        self.code = ""
        self.name = ""

    def __str__(self):
        return "Code: %s, Name: %s" % (self.code, self.name)


class DepthMktDataDescription(Object):
    def __init__(self):
        self.exchange = ""
        self.secType = ""
        self.listingExch = ""
        self.serviceDataType = ""
        self.aggGroup = UNSET_INTEGER

    def __str__(self):
        if (self.aggGroup!= UNSET_INTEGER):
            aggGroup = self.aggGroup
        else:
            aggGroup = ""
        return "Exchange: %s, SecType: %s, ListingExchange: %s, ServiceDataType: %s, AggGroup: %s, " % (self.exchange, self.secType, 
            self.listingExch,self.serviceDataType, ibapi.utils.intMaxString(aggGroup))


class SmartComponent(Object):
    def __init__(self):
        self.bitNumber = 0
        self.exchange = ""
        self.exchangeLetter = ""

    def __str__(self):
        return "BitNumber: %d, Exchange: %s, ExchangeLetter: %s" % (self.bitNumber, self.exchange, self.exchangeLetter)


class TickAttrib(Object):
    def __init__(self):
        self.canAutoExecute = False
        self.pastLimit = False
        self.preOpen = False

    def __str__(self):
        return "CanAutoExecute: %d, PastLimit: %d, PreOpen: %d" % (self.canAutoExecute, self.pastLimit, self.preOpen)


class TickAttribBidAsk(Object):
    def __init__(self):
        self.bidPastLow = False
        self.askPastHigh = False

    def __str__(self):
        return "BidPastLow: %d, AskPastHigh: %d" % (self.bidPastLow, self.askPastHigh)


class TickAttribLast(Object):
    def __init__(self):
        self.pastLimit = False
        self.unreported = False

    def __str__(self):
        return "PastLimit: %d, Unreported: %d" % (self.pastLimit, self.unreported)


class FamilyCode(Object):
    def __init__(self):
        self.accountID = ""
        self.familyCodeStr = ""

    def __str__(self):
        return "AccountId: %s, FamilyCodeStr: %s" % (self.accountID, self.familyCodeStr)


class PriceIncrement(Object):
    def __init__(self):
        self.lowEdge = 0.
        self.increment = 0.

    def __str__(self):
        return "LowEdge: %s, Increment: %s" % (ibapi.utils.floatMaxString(self.lowEdge), ibapi.utils.floatMaxString(self.increment))


class HistoricalTick(Object):
    def __init__(self):
        self.time = 0
        self.price = 0.
        self.size = UNSET_DECIMAL

    def __str__(self):
        return "Time: %s, Price: %s, Size: %s" % (ibapi.utils.intMaxString(self.time), ibapi.utils.floatMaxString(self.price), ibapi.utils.decimalMaxString(self.size))


class HistoricalTickBidAsk(Object):
    def __init__(self):
        self.time = 0
        self.tickAttribBidAsk = TickAttribBidAsk()
        self.priceBid = 0.
        self.priceAsk = 0.
        self.sizeBid = UNSET_DECIMAL
        self.sizeAsk = UNSET_DECIMAL

    def __str__(self):
        return "Time: %s, TickAttriBidAsk: %s, PriceBid: %s, PriceAsk: %s, SizeBid: %s, SizeAsk: %s" % (ibapi.utils.intMaxString(self.time), self.tickAttribBidAsk, 
            ibapi.utils.floatMaxString(self.priceBid), ibapi.utils.floatMaxString(self.priceAsk), 
            ibapi.utils.decimalMaxString(self.sizeBid), ibapi.utils.decimalMaxString(self.sizeAsk))


class HistoricalTickLast(Object):
    def __init__(self):
        self.time = 0
        self.tickAttribLast = TickAttribLast()
        self.price = 0.
        self.size = UNSET_DECIMAL
        self.exchange = ""
        self.specialConditions = ""

    def __str__(self):
        return "Time: %s, TickAttribLast: %s, Price: %s, Size: %s, Exchange: %s, SpecialConditions: %s" % (ibapi.utils.intMaxString(self.time), self.tickAttribLast, 
            ibapi.utils.floatMaxString(self.price), ibapi.utils.decimalMaxString(self.size), self.exchange, self.specialConditions)

class HistoricalSession(Object):
    def __init__(self):
        self.startDateTime = ""
        self.endDateTime = ""
        self.refDate = ""

    def __str__(self):
        return "Start: %s, End: %s, Ref Date: %s" % (self.startDateTime, self.endDateTime, self.refDate)

class WshEventData(Object):
    def __init__(self):
        self.conId = UNSET_INTEGER
        self.filter = ""
        self.fillWatchlist = False
        self.fillPortfolio = False
        self.fillCompetitors = False
        self.startDate = ""
        self.endDate = ""
        self.totalLimit = UNSET_INTEGER

    def __str__(self):
        return "WshEventData. ConId: %s, Filter: %s, Fill Watchlist: %d, Fill Portfolio: %d, Fill Competitors: %d" % (ibapi.utils.intMaxString(self.conId), 
            self.filter, self.fillWatchlist, self.fillPortfolio, self.fillCompetitors)

