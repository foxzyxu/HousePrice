import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import Ridge


HP_train = pd.read_csv('train.csv')
HP_train['MSSubClass'] = HP_train['MSSubClass'].astype(str)
#print(HP_train.columns.values)
#特征相关性分析(计算变量与房价的相关性)
corrmat = HP_train.corr()
faetures =[x for x in corrmat.columns]
salecorr = [y for y in corrmat['SalePrice']]
spr = pd.DataFrame()
spr['feature'] = faetures
spr['salecorr'] = salecorr
spr = spr.sort_values(by = ['salecorr'], ascending = True, axis = 0)
#f, ax = plt.subplots(figsize=(8, 24))
#sns.barplot(x = 'salecorr', y = 'feature', data = spr)
##两个特征的相关性，用于去除相似的特征
#sns.set()
#f, ax = plt.subplots(figsize=(16, 12))
#sns.heatmap(corrmat, vmax=.8, square=True);
#与售价相关性较强的特征
cols = ['SalePrice', 'KitchenAbvGr', 'EnclosedPorch', 'OverallQual', 
        'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
#sns.set()
#sns.pairplot(HP_train[cols], height = 2)
#plt.show()
#缺失值处理，选取cols中的特征
#print(HP_train.info())
#处理缺失值过多的
missing = HP_train.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([missing], axis=1, keys=['count'])
#print(missing_data)
HP_train = HP_train.drop((missing_data[missing_data['count'] > 1]).index,1)
HP_train = HP_train.drop(HP_train.loc[HP_train['Electrical'].isnull()].index)
#print(HP_train.isnull().sum().max())
#删除离群点
#data = pd.concat([HP_train['SalePrice'], HP_train['GrLivArea']], axis=1)
#data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
HP_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
HP_train = HP_train.drop(HP_train[HP_train['Id'] == 1299].index)
HP_train = HP_train.drop(HP_train[HP_train['Id'] == 524].index)
quantity = [x for x in HP_train.columns if HP_train.dtypes[x] != 'object']
quality = [x for x in HP_train.columns if HP_train.dtypes[x] == 'object']
#哑变量化定性特征
HP_train = pd.get_dummies(HP_train)
#标准化(不必要)和正态化
HP_train['SalePrice'] = np.log(HP_train['SalePrice'])
#sns.set()
#sns.distplot(HP_train['SalePrice'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(HP_train['SalePrice'], plot=plt)
#含有较多0值的数据如何正态化
HP_train['GrLivArea'] = np.log(HP_train['GrLivArea'])
HP_train['TotalBsmtSF'] = np.log(HP_train['TotalBsmtSF']+1)
#sns.set()
#sns.distplot(HP_train[HP_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
#fig = plt.figure()
#res = stats.probplot(HP_train[HP_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
HP_train['LotArea'] = np.log(HP_train['LotArea'])
HP_train['BsmtFinSF1'] = np.log(HP_train['BsmtFinSF1']+1)
HP_train['BsmtFinSF2'] = np.log(HP_train['BsmtFinSF2']+1)
HP_train['BsmtUnfSF'] = np.log(HP_train['BsmtUnfSF']+1)
HP_train['1stFlrSF'] = np.log(HP_train['1stFlrSF'])
HP_train['2ndFlrSF'] = np.log(HP_train['2ndFlrSF']+1)
HP_train['LowQualFinSF'] = np.log(HP_train['LowQualFinSF']+1)
HP_train['GarageArea'] = np.log(HP_train['GarageArea']+1)
HP_train['WoodDeckSF'] = np.log(HP_train['WoodDeckSF']+1)
HP_train['OpenPorchSF'] = np.log(HP_train['OpenPorchSF']+1)
HP_train['EnclosedPorch'] = np.log(HP_train['EnclosedPorch']+1)
HP_train['3SsnPorch'] = np.log(HP_train['3SsnPorch']+1)
HP_train['ScreenPorch'] = np.log(HP_train['ScreenPorch']+1)
HP_train['PoolArea'] = np.log(HP_train['PoolArea']+1)
HP_train['MiscVal'] = np.log(HP_train['MiscVal']+1)
#得到训练集
Y_train = HP_train['SalePrice'].values
X_faetures = [x for x in HP_train.columns]
X_faetures.remove('Id')
X_faetures.remove('SalePrice')
#补充缺失的特征集，原因见测试集数据处理
X_train = HP_train[X_faetures]
X_train.insert(33,'MSSubClass_150',[0 for i in range(0,1457)])
#X_faetures = [x for x in X_train.columns]
X_train = X_train.values

#岭回归模型
clf = Ridge()
clf.fit(X_train, Y_train)

#处理待测数据
HP_test = pd.read_csv('test.csv')
HP_test['MSSubClass'] = HP_test['MSSubClass'].astype(str)
Test_fatures = quantity + quality
Test_fatures.remove('SalePrice')
X_test = HP_test[Test_fatures]
#print(X_test.info())
#Test_missing = X_test.isnull().sum().sort_values(ascending=False)
X_test.MSZoning[X_test.MSZoning.isnull()] = X_test.MSZoning.dropna().mode().values
X_test.Utilities[X_test.Utilities.isnull()] = X_test.Utilities.dropna().mode().values
X_test.Functional[X_test.Functional.isnull()] = X_test.Functional.dropna().mode().values
X_test.SaleType[X_test.SaleType.isnull()] = X_test.SaleType.dropna().mode().values
X_test.KitchenQual[X_test.KitchenQual.isnull()] = X_test.KitchenQual.dropna().mode().values
X_test.BsmtFullBath[X_test.BsmtFullBath.isnull()] = X_test.BsmtFullBath.dropna().mode().values
X_test.Exterior1st[X_test.Exterior1st.isnull()] = X_test.Exterior1st.dropna().mode().values
X_test.Exterior2nd[X_test.Exterior2nd.isnull()] = X_test.Exterior2nd.dropna().mode().values
X_test.GarageCars[X_test.GarageCars.isnull()] = X_test.GarageCars.dropna().mode().values
X_test['BsmtHalfBath'] = X_test['BsmtHalfBath'].fillna(0)
#根据某个相关变量改变某个值
#Half_missing = X_test['BsmtHalfBath'].isnull()
#X_test.BsmtHalfBath[Half_missing] = (X_test.BsmtFullBath[Half_missing] * 0.5)
#BsmtSF = ['BsmtFinSF1', 'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF']
#sns.pairplot(X_test[BsmtSF].dropna())
X_test['BsmtFinSF2'] = X_test['BsmtFinSF2'].fillna(0)
X_test.BsmtFinSF1[X_test.BsmtFinSF1.isnull()] = X_test.BsmtFinSF1.dropna().mean()
X_test.BsmtUnfSF[X_test.BsmtUnfSF.isnull()] = X_test.BsmtUnfSF.dropna().mean()
X_test.TotalBsmtSF[X_test.TotalBsmtSF.isnull()] = X_test.TotalBsmtSF.dropna().mean()
X_test.GarageArea[X_test.GarageArea.isnull()] = X_test.GarageArea.dropna().mean()
#Test_missing = X_test.isnull().sum().sort_values(ascending=False)
Y_Id = X_test['Id']
X_test.drop(['Id'], axis=1,inplace=True)
X_test = pd.get_dummies(X_test)
X_test['GrLivArea'] = np.log(X_test['GrLivArea'])
X_test['TotalBsmtSF'] = np.log(X_test['TotalBsmtSF']+1)
X_test['LotArea'] = np.log(X_test['LotArea'])
X_test['BsmtFinSF1'] = np.log(X_test['BsmtFinSF1']+1)
X_test['BsmtFinSF2'] = np.log(X_test['BsmtFinSF2']+1)
X_test['BsmtUnfSF'] = np.log(X_test['BsmtUnfSF']+1)
X_test['1stFlrSF'] = np.log(X_test['1stFlrSF'])
X_test['2ndFlrSF'] = np.log(X_test['2ndFlrSF']+1)
X_test['LowQualFinSF'] = np.log(X_test['LowQualFinSF']+1)
X_test['GarageArea'] = np.log(X_test['GarageArea']+1)
X_test['WoodDeckSF'] = np.log(X_test['WoodDeckSF']+1)
X_test['OpenPorchSF'] = np.log(X_test['OpenPorchSF']+1)
X_test['EnclosedPorch'] = np.log(X_test['EnclosedPorch']+1)
X_test['3SsnPorch'] = np.log(X_test['3SsnPorch']+1)
X_test['ScreenPorch'] = np.log(X_test['ScreenPorch']+1)
X_test['PoolArea'] = np.log(X_test['PoolArea']+1)
X_test['MiscVal'] = np.log(X_test['MiscVal']+1)
#发现亚量化后特征值数量不对
Test_fatures = [x for x in X_test.columns]
Share_features = list(set(X_faetures) & set(Test_fatures))
Test_needfeatures = list(set(X_faetures) - set(Share_features))
Train_needfeatures = list(set(Test_fatures) - set(Share_features))
#数值化测试集
X_test.insert(119,'HouseStyle_2.5Fin',[0 for i in range(0,1459)])
X_test.insert(111,'Condition2_RRNn',[0 for i in range(0,1459)])
X_test.insert(132,'RoofMatl_Membran',[0 for i in range(0,1459)])
X_test.insert(111,'Condition2_RRAn',[0 for i in range(0,1459)])
X_test.insert(183,'Heating_OthW',[0 for i in range(0,1459)])
X_test.insert(146,'Exterior1st_Stone',[0 for i in range(0,1459)])
X_test.insert(134,'RoofMatl_Roll',[0 for i in range(0,1459)])
X_test.insert(64,'Utilities_NoSeWa',[0 for i in range(0,1459)])
X_test.insert(146,'Exterior1st_ImStucc',[0 for i in range(0,1459)])
X_test.insert(199,'Electrical_Mix',[0 for i in range(0,1459)])
X_test.insert(184,'Heating_Floor',[0 for i in range(0,1459)])
X_test.insert(112,'Condition2_RRAe',[0 for i in range(0,1459)])
X_test.insert(136,'RoofMatl_Metal',[0 for i in range(0,1459)])
X_test.insert(165,'Exterior2nd_Other',[0 for i in range(0,1459)])
#Test_fatures = [x for x in X_test.columns]
X_test = X_test.values
##出入数据进行预测
Predicted_Y = clf.predict(X_test)
HousePrice_submission = pd.DataFrame()
HousePrice_submission['Id'] = Y_Id.values
HousePrice_submission['SalePrice'] = np.exp(Predicted_Y)
HousePrice_submission.to_csv('HousePrice_submission.csv', index = False)