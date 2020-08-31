#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# author: Xiaoxuan Bi
"""
    XiaoxuanBi_code.py
    描述：基金持仓的行业时间序列、面板数据规整。基于持仓数据整体股票型基金风格归因分析
    输入：
        - 股票持仓数据：'CHINAMUTUALFUNDSTOCKPORTFOLIO.csv'
    输出：
        - 分析图表：因子收益柱状图，因子归因时序图，两张因子相关矩阵图
        - 格式：png 
        - 数据文件默认命名格式 'factor_bar.png' 'factor_time.png' '03_09_Corr.png' '09_11_Corr.png'
"""

"""
Packages used：
"""
import pandas as pd
import time 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jqfactor import get_factor_values

class Segment:
    def __init__(self, stockIdList, stkValueToNavList, time, factorYield = None):
        assert(len(stockIdList) == len(stkValueToNavList))
        self.segment = {}#Each time segment has a stockpool, with a bunch of Stock object
        self.stockIdList =  stockIdList #list
        self.stkValueToNavList = stkValueToNavList #list
        self.time = time #float
        self.factorYield = factorYield #array
        self.segment_df = None #DataFrame
        self.fYield = None #array
        
        for i in range(len(stockIdList)):
            stock = Stock(time,stockIdList[i],stkValueToNavList[i])
            self.segment[stockIdList[i]] = stock
        self.createSegmentDf()
    
    def createSegmentDf(self):
        self.segment_df = pd.DataFrame.from_dict(self.segment,orient = 'index')
    
    def getTime(self):
        return self.time
    
    def getStockList(self):
        return self.stockIdList
    
    def getStkValueToNavList(self):
        return self.stkValueToNavList
    
    def getFYield(self):
        return self.fYield
    
    def getSegment(self):
        return self.segment
    
    def getSegmentDf(self):
        return self.segment_df
    
    def setFactorYield(self,factorYield):
        self.factorYield = factorYield
        
    def setStockIdList(self,stockIdList):
        self.stockIdList = stockIdList
        
    def setSegment(self,stockIdList):
        seg_df = self.segment_df
        self.segment_df = seg_df[seg_df.index.isin(stockIdList)]
        self.setStkValueToNavList()
        
    def setStkValueToNavList(self):
        li = []
        for stock in self.segment_df:
            seg_value = self.segment_df[stock].values
            for i in seg_value:
                li.append(i.stkValueToNav)
        self.stkValueToNavList = li
    
    def setYield(self,f):
        self.fYield = f

class Stock:
    def __init__(self, time, name, stkValueToNav, factor = None):
        self.time = time
        self.name = name
        self.stkValueToNav = stkValueToNav
        self.factor = stkValueToNav


"""
任务：读取数据，数据预处理：截取需要的时间，处理缺失值，统一股票代码名称
"""
df = pd.read_csv("CHINAMUTUALFUNDSTOCKPORTFOLIO.csv")
df = df[(df['F_PRT_ENDDATE']>20050101)]
df = df.drop(df[df['F_PRT_STKVALUETONAV'].isnull()].index)
foundList = df.drop_duplicates(['S_INFO_WINDCODE'])['S_INFO_WINDCODE']
df['S_INFO_STOCKWINDCODE'] = df['S_INFO_STOCKWINDCODE'].str.replace('SH','XSHG').str.replace('SZ','XSHE')


"""
任务：面板数据规整，创建基金字典，
面板数据格式：
- 以每一个基金代码作为key, 创建基金字典foundDict。
- 每个基金下存放 segment 对象，存放该基金在记录时间节点上投资组合信息 
- segment对象储存time，stockList，stkValueToNavList,以及后面计算出来的因子暴露值
"""
foundDict = {}
for found in foundList:
    foundRecord = df[df['S_INFO_WINDCODE'] == found]
    timeList = foundRecord.drop_duplicates(['F_PRT_ENDDATE'])['F_PRT_ENDDATE']
    foundDict[found] = {}
    for time in timeList:
        segementData = foundRecord[foundRecord['F_PRT_ENDDATE'] == time]
        stockIdList =  list(segementData['S_INFO_STOCKWINDCODE'])
        stkValueToNavList = list(segementData['F_PRT_STKVALUETONAV'])
        foundDict[found][time] = Segment(stockIdList, stkValueToNavList, time) #得到：一个基金在一个时间截面的数据

"""
用于获取聚宽因子库里风格因子暴露数组： nX10 array
"""
def getFactor(time,stockList): 
    #this function is used for access factor dict form JoinQuant factor databases for our selected stocks
    stockDict = {}
    for stock in stockDict:
        factor_data = get_factor_values(securities=stock, factors=["beta","book_to_price_ratio","momentum","residual_volatility","non_linear_size" ,"liquidity" ,"size","earnings_yield","growth","leverage"], start_date=time, count = 1)
        print(factor_data)
        data_frame = pd.DataFrame()

        for key, value in factor_data.items():
            if len(data_frame) == 0:
                data_frame = value
            else:
                df1 = value.fillna(method = 'bfill') 
                data_frame = pd.merge(data_frame,df1, left_index=True, right_index = True)
        data_frame.columns = ["book_to_price_ratio","growth","leverage","size", "earnings_yield","momentum","liquidity","residual_volatility","beta","non_linear_size" ]
        
        stockDict[stock] = data_frame
    
    return stockDict        

"""
获得权重Weight数组： nXn array
"""
def getWeight(a):
    #this function is used for getting weight for our factor, 
    try:
        a_inv = np.linalg.inv(a)
    except:
        return None   
        pass
    a_T = a.T
    try:
        w = np.dot(np.linalg.inv(np.dot(a_T,a)),a_T)
    except:
        return None   
        pass
    return w

"""
Transform dictionary factor to np.array, 
return a nX10 array (n is number of stocks, 10 represent the number of our fators)
"""
def processFactor(seg_factor_data):

    if seg_factor_data is not None:
        df_temp = pd.DataFrame()
        seg_list = []
        for key, value in seg_factor_data.items():
            seg_list.append(value.iloc[0].values)
            
        ##create a matrix
        seg_array = np.array(seg_list)
        seg_df = pd.DataFrame(seg_array)
        
        ##fill N/A in matrix, we fisrt change array to DataFrame, use pandas to fillna
        if 10 in seg_df.isna().sum().values:
            return None
        else: 
            seg_df = seg_df.fillna(method = "bfill")
            seg_array = seg_df.values.T#change DF back to array
            return seg_array

"""
获取因子收益率数组 10X1
f = inv(X_TWX)X_TWR
"""
def get_f(x_a, w_a,r_a):
    x_t = x_a.T
    f = np.linalg.inv((x_t.dot(w_a)).dot(x_a)).dot((x_t.dot(w_a)).dot(r_a))
    return f


"""
Main function
任务：针对每个基金各个时间截面，计算基金股票池收益归因矩阵，并将结果储存在segment object 'fYield' 的变量里
"""
from jqfactor import get_factor_values
count = 0
for founder in foundDict:
    segements = foundDict[founder]
    new_segements = {}
    if founder not in new_foundDict:
        new_foundDict[founder] = None
    
    for seg in segements:
        
        seg_time = segements[seg].time
        seg_stockList = segements[seg].stockIdList
        seg_r_time = pd.to_datetime(str(seg_time), format='%Y%m%d.0').date()
        oc_data = []#list used for storing 收益率
        stock_used = []#list used for storing stocks that we can successfully get 收益率, weight and factor array
        
        #计算股票收益率矩阵 nX1
        for stock in seg_stockList:
            try:
                oc_stock_data = get_price(stock, start_date=seg_r_time, end_date=seg_r_time, frequency='daily', fields=['open', 'close'],panel = False)   
            except:
                seg_stockList.remove(stock)
                continue
            if len(oc_stock_data.index)!=0:
                market = (oc_stock_data['close']/oc_stock_data['open'])- 1.0
                market = market.fillna(0).values
                oc_data.append(market)
            else:
                seg_stockList.remove(stock)
                
        #if no yield data for stocklist, jump into next loop        
        if len(oc_data) == 0 :
            continue
        else:
            if len(seg_stockList) != len(oc_data):#对齐数据
                continue
                
            else:   
                #get 单位stock i 在 factor f 上的风格暴露。转换字典型数据为 nX10 数组
                seg_factor_data = get_factor_values(securities=seg_stockList, factors=["beta","book_to_price_ratio","momentum","residual_volatility","non_linear_size" ,"liquidity" ,"size","earnings_yield","growth","leverage"], start_date=seg_r_time, count = 1)
                factor_array = processFactor(seg_factor_data)
                if factor_array is None:
                    continue
                    
                #get Weight array 10Xn
                weight = getWeight(factor_array)
                oc_array = np.array(oc_data)
                if weight is not None:
                    if weight.size !=0 and factor_array.size !=0 and oc_array.size != 0:
                        try:
                            #得到单位股票在10个风格因子上的风格暴露，10X1
                            f_array = get_f(factor_array,weight,oc_array) 
                        except:
                            continue
                        if f_array.size !=0:
                            #save factor yield array into our segments object,自动更新segment stocklist，stklist
                            segements[seg].setFactorYield(f_array)
                            segements[seg].setStockIdList(seg_stockList)
                            segements[seg].setSegment(seg_stockList)            
                else:
                    continue

        #从基于持仓数据的面板数据获得投资比例 nX1
        seg_stkValueToNavList = segements[seg].stkValueToNavList
        
        if len(seg_stkValueToNavList) == f_array.shape[0]:
            ##get stock i 在 factor f 上的风格暴露 factor_array_share nX1
            stk_array = (np.array(seg_stkValueToNavList)/100).T
            stk_reshape = np.reshape(stk_array,(len(stk_array),1))#投资比例
            factor_array_share = factor_array.dot(stk_reshape)
            ##get 基金收益归因矩阵, nX10, 每一列代表这个基金在一种风格下的收益
            r_array = factor_array_share.dot(f_array.T)
            r_df = pd.DataFrame(r_array)
            segements[seg].setYield(r_array)


"""
计算各因子因子暴露的平均值，涵盖所有基金以及所有时间截面，储存平均矩阵到"Final_f.txt"
"""
count = 0
import numpy as np

def filter_fun(x):#this function is used for checking whether nan in ndarray
    nanIn = False
    for i in x:
        if i!= i:
            return True
        else:
            continue
    return nanIn

for founder in foundDict:
    segements = foundDict[founder]
    for time in segements: #每个founder在每个segment有一个 nX10 fYield
        seg_df = segements[time].fYield
        if seg_df is not None:
            for each_f in seg_df:
                if filter_fun(each_f) is False:
                    if seg_sum is None:
                        seg_sum = each_f
                    else:
                        seg_sum = np.vstack((seg_sum,each_f))                               

count_arry = np.array([seg_sum.shape[0]]*10)
Final_f = seg_sum.sum(axis=0)/count_arry

#保存
np.savetxt("Final_f.txt",Final_f)

#读取: 验证可读取
b =numpy.loadtxt("Final_f.txt", delimiter=',')

"""
计算不同时间截面的平均因子暴露，储存到字典'timeSeriesDict.txt'
"""
def calAveDict(timeDict):
    for time in timeDict:
        seg_array = timeDict[time]
        if seg_array is not None:
            count_array = np.array([seg_array.shape[0]]*10)
            Final_f = seg_array.sum(axis=0)/count_array
        temp_time_dict[time] = Final_f

# 将每个时间截面归因矩阵以vstack形式合并，存于timeDict
timeDict = {}
for founder in foundDict:
    segements = foundDict[founder]
    for time in segements:#每个founder在每个segment有一个fYield
        seg_df = segements[time].fYield
        if seg_df is not None:
            row = 0
            seg_df_temp = None
            for each_f in seg_df:
                if filter_fun(each_f) is False:#check if it is nan in ndarray
                    if seg_df_temp is None:
                        seg_df_temp = each_f
                    else:
                        seg_df_temp = np.vstack((seg_df_temp,each_f))
                else:
                    continue
            if seg_df_temp is not None:
                seg_df = seg_df_temp
            else:
                continue
            if time not in timeDict:
                timeDict[time] = seg_df
            else:
                timeDict[time] = np.vstack((timeDict[time],seg_df))

temp_time_dict = {}
calAveDict(timeDict)

#保存
file = open('timeSeriesDict.txt','w')
file.write(str(temp_time_dict))
file.close()
 
#读取: 验证可读取
f = open('timeSeriesDict.txt','r')
a = f.read()
dict_name = eval(a)
f.close()


"""
存储面板数据
"""
csv_dict = {}
csv_dict["S_INFO_WINDCODE"] = []
csv_dict["F_PRT_ENDDATE"] = []
csv_dict["S_INFO_STOCKWINDCODE_LIST"] = []
csv_dict["F_PRT_STKVALUETONAV_LIST"] = []
csv_dict["BARRA_FACTOR_LIST"] = []

def calAvgArray(array):
    avg = np.mean(array,axis = 0)
    return avg

for founder in foundDict:
    segements = foundDict[founder]
    for time in segements:#每个founder在每个segment有一个fYield
        founder = founder
        csv_dict["S_INFO_WINDCODE"].append(founder)
        time = time
        csv_dict["F_PRT_ENDDATE"].append(time)
        stocklist = segements[time].getStockList()
        csv_dict["S_INFO_STOCKWINDCODE_LIST"].append(stocklist)
        StkValueToNavList = segements[time].stkValueToNavList
        csv_dict["F_PRT_STKVALUETONAV_LIST"].append(StkValueToNavList)
        try:
            fYield = segements[time].fYield
            csv_dict["BARRA_FACTOR_LIST"].append(calAvgArray(fYield))
        except:
            csv_dict["BARRA_FACTOR_LIST"].append(None)
csv_df= pd.DataFrame.from_dict(csv_dict,orient='columns')
csv_df.to_csv('Panel.csv')
csv_df.to_excel('Panel.xlsx')


"""
绘制长期因子暴露柱状图"factor_bar"
"""
factorList = ["book_to_price_ratio","growth","leverage","size", "earnings_yield","momentum","liquidity","residual_volatility","beta","non_linear_size" ]
factor = ("book_to_price_ratio","growth","leverage","size", "earnings_yield","momentum","liquidity","residual_volatility","beta","non_linear_size" )

matplotlib.style.use('ggplot')

index = np.arange(10)
df1 = pd.Series(b)
plt.figure(figsize = (15,8))
p1 = plt.bar(index, df1)
plt.axhline(0,color = 'k')
plt.xticks(index,factor, size = 12)
plt.title('调整后长期风格因子暴露',size = 20)
plt.savefig('factor_bar.png')

"""
去极值
"""
#去极值
def delextremum(array):
    sigma = array.std()
    mu = array.mean()
    array[array > mu + 3*sigma] = mu + 3*sigma
    array[array < mu - 3*sigma] = mu - 3*sigma
    array[array >0.2] = 0
    array[array < -0.2] = 0
    return array

"""
绘制时间序列下因子暴露变化折线图'factor_time.png'
"""
#plot timeSeries factor plot
time_df = pd.DataFrame(temp_time_dict)
time_df.index = factorList
time_df.columns = pd.to_datetime(time_df.columns.astype(str), format='%Y%m%d.0')

###prepare to draw timeserise plot
btpr = delextremum(time_df.loc['book_to_price_ratio'].values).tolist()
growth = delextremum(time_df.loc['growth'].values).tolist()
leverage = delextremum(time_df.loc['leverage'].values).tolist()
size = delextremum(time_df.loc['size'].values).tolist()
ey = delextremum(time_df.loc['earnings_yield'].values).tolist()
momentum = delextremum(time_df.loc['momentum'].values).tolist()
liquidity = delextremum(time_df.loc['liquidity'].values).tolist()
rv = delextremum(time_df.loc['residual_volatility'].values).tolist()
beta = delextremum(time_df.loc['beta'].values).tolist()
nls = delextremum(time_df.loc['non_linear_size'].values).tolist()

data = {
    'book_to_price_ratio':btpr,
    'growth': growth,
    'leverage':leverage,
    'size': size,
    'earnings_yield':ey,
    'momentum':momentum,
    'liquidity':liquidity,
    'residual_volatility':rv,
    'beta':beta,
    'non_linear_size':nls
       }

data_f = pd.DataFrame(data,index=time_df.columns)
plt.figure(figsize = (15,8))
#移动平均降噪
data_f.rolling(2).mean().plot(figsize = (15,8))
plt.ylim(-0.15,0.1)
plt.title('长期来看，风格因子暴露变化',size = 20)
plt.savefig('factor_time.png')

"""
绘制2003-2009 以及2009-2010年因子相关矩阵。
"""
data_f1 = data_f[data_f.index < "2009-01-01"]
data_f2 = data_f[data_f.index >="2009-01-01"]
plt.figure()
data_f1.corr().style.background_gradient(cmap='coolwarm')
data_f2.corr().style.background_gradient(cmap='coolwarm')