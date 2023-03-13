import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 使用 ggplot 画图风格
plt.style.use('ggplot')

# 加载训练集数据
data = pd.read_csv('../data/train.csv', index_col=0)

# 打印数据的基本信息（列的信息，数据类型等）
data.info()

# 统计数值类型(numerical)特征名，保存为列表
numeric = [f for f in data.drop(['SalePrice'], axis=1).columns
           if data.drop(['SalePrice'], axis=1).dtypes[f] != 'object']
# 统计类别类型（category）特征名，保存为列表
category = [f for f in data.drop(['SalePrice'], axis=1).columns
            if data.drop(['SalePrice'], axis=1).dtypes[f] == 'object']
# 输出数值类型(numerical)特征个数，类型（category）特征个数
print("numeric: {}, category: {}".format(len(numeric), len(category)))

# 统计每个特征的缺失数
missing = data.isnull().sum()
# 将缺失数从大到小排序
missing.sort_values(inplace=True, ascending=False)
# 选取缺失数大于0的特征
missing = missing[missing > 0]
# 保存含缺失值特征的类型
types = data[missing.index].dtypes
# 计算缺失值百分比
percent = missing / data.shape[0]
# 将缺失值信息整合
missing_data = pd.concat([missing, percent, types], axis=1,
                         keys=['Total', 'Percent', 'Types'])
# 输出缺失值信息
print(missing_data)

# 对缺失值超过15%的特征进行删除
data.drop(missing_data[missing_data['Percent'] > 0.15].
          index, axis=1, inplace=True)

# 用Electrical列的众数去填充缺失值
data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=True)

cols1 = ['GarageFinish', 'GarageQual', 'GarageType', 'GarageCond',
         'BsmtFinType2', 'BsmtExposure',
         'BsmtFinType1', 'BsmtQual', 'BsmtCond', 'MasVnrType']
# 依次便利cols1中的特征，对应缺失值用‘None’填充
for col in cols1:
    data[col].fillna('None', inplace=True)

# 由data_description里面的内容的可知，对于数值型数据，如MasVnrArea
# 表示砖石贴面面积，如果一个房子本身没有砖石贴面，则缺失值就用0来填补
data['MasVnrArea'].fillna(0, inplace=True)

year_map = pd.concat(pd.Series('YearGroup' + str(i + 1),
                               index=range(1871 + i * 20, 1891 + i * 20)) for i in range(0, 7))
data['GarageYrBlt'] = data['GarageYrBlt'].map(year_map)
# 对时间缺失值用‘None’填充
data['GarageYrBlt'] = data['GarageYrBlt'].fillna('None')

# 常看是否还有空值，最终结果为空即不存在
data.isnull().sum()[data.isnull().sum() > 0]

data['SalePrice'].describe()

sns.distplot(data['SalePrice'], hist=True, kde=True)

# 偏度skewness and 峰度kurtosis计算
# 偏度值离0越远，则越偏
print("Skewness: %f" % data['SalePrice'].skew())
print("Kurtosis: %f" % data['SalePrice'].kurt())

data['SalePrice'] = np.log1p(data["SalePrice"])
# 画出概率密度图
sns.distplot(data['SalePrice'], hist=True, kde=True)

print("Skewness: %f" % data['SalePrice'].skew())
print("Kurtosis: %f" % data['SalePrice'].kurt())

data = data[np.abs(data['SalePrice'] - data['SalePrice'].mean()) <= (3 * data['SalePrice'].std())]

res = stats.probplot(data['SalePrice'], plot=plt)

# 计算所有数值型特征与房价的相关系数
corrmat = data.corr()
# 计算与房价相关性大于0.5的特征个数
k = len(corrmat[corrmat['SalePrice'] > 0.5].index)
# 获取前k个重要的特征名
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index.tolist()
# 计算该k个特征的相关系数
cm = data[cols].corr()
# 画出可视化热图
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, square=True)

sns.set()
area = ['SalePrice', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageArea']
sns.pairplot(data[area], size=2.5)

data.drop(data[data['TotalBsmtSF'] > 4000].index, inplace=True)
data.drop(data[data['1stFlrSF'] > 4000].index, inplace=True)
data.drop(data[data['GrLivArea'] > 4000].index, inplace=True)
data.drop(data[data['GarageArea'] > 1240].index, inplace=True)

print("Skewness: %f" % data['TotalBsmtSF'].skew())
print("Kurtosis: %f" % data['TotalBsmtSF'].kurt())
print("Skewness: %f" % data['1stFlrSF'].skew())
print("Kurtosis: %f" % data['1stFlrSF'].kurt())
print("Skewness: %f" % data['GrLivArea'].skew())
print("Kurtosis: %f" % data['GrLivArea'].kurt())
print("Skewness: %f" % data['GarageArea'].skew())
print("Kurtosis: %f" % data['GarageArea'].kurt())

data['ln_1stFlrSF'] = np.log1p(data["1stFlrSF"])
data['ln_GrLivArea'] = np.log1p(data["GrLivArea"])
print("Skewness: %f" % data['ln_1stFlrSF'].skew())
print("Kurtosis: %f" % data['ln_1stFlrSF'].kurt())
print("Skewness: %f" % data['ln_GrLivArea'].skew())
print("Kurtosis: %f" % data['ln_GrLivArea'].kurt())

sns.countplot(x='MoSold', data=data)

numeric = [f for f in data.drop(['SalePrice'], axis=1).columns
           if data.drop(['SalePrice'], axis=1).dtypes[f] != 'object']
category = [f for f in data.drop(['SalePrice'], axis=1).columns
            if data.drop(['SalePrice'], axis=1).dtypes[f] == 'object']
# 输出数值类型(numerical)特征个数，类型（category）特征个数
print("numeric: {}, category: {}".format(len(numeric), len(category)))


# 定义方差函数，返回p-value值，其值越小对应特征越重要
def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = category
    pvals = []
    for c in category:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        # stats.f_onewaym模块包用于计算p-value
        pval = stats.f_oneway(*samples)[1]
        pvals.append(pval)
    anv['pval'] = pvals
    # 将特征根据p-valu排序
    return anv.sort_values('pval')


# 将data带入定义的方差函数
a = anova(data)

# 选择相关性大于0.5的重要数值型特征
df = data[cols]
# 将 1stFlrSF 和 GrLivArea 特征ln(x+1)转化
df["1stFlrSF"] = data['ln_1stFlrSF']
df["GrLivArea"] = data['ln_GrLivArea']
# 将时间特征离散化，即没20年分段
df['YearBuilt'] = df['YearBuilt'].map(year_map)
df['YearRemodAdd'] = df['YearRemodAdd'].map(year_map)
# 对非线性特征 MoSold one-hot编码
month = pd.get_dummies(data['MoSold'], prefix='MoSold')
# 合并特征
df = pd.concat([df, month], axis=1)

# 对于类别型数据，跟据方差分析，选取排名重要的25个特征：
features = a['feature'].tolist()[0:25]
# 合并特征
df = pd.concat([df, data[features]], axis=1)

# 查看特征维度
df.shape

# 特征融合
# 将面积特征相加，构建总面积特征
df["TotalHouse"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]
df["TotalArea"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"] + data["GarageArea"]
# 画出其回归图
sns.jointplot(x=df["TotalHouse"], y=df['SalePrice'], data=df, kind="reg")
sns.jointplot(x=df["TotalArea"], y=df['SalePrice'], data=df, kind="reg")

df.drop(df[df['TotalHouse'] > 6000].index, inplace=True)
df.drop(df[df['TotalArea'] > 6500].index, inplace=True)

# 继续融合特征
# 将部分相关联的特征进行相加或相乘
df["+_TotalHouse_OverallQual"] = df["TotalHouse"] * data["OverallQual"]
df["+_GrLivArea_OverallQual"] = data["GrLivArea"] * data["OverallQual"]
df["+_BsmtFinSF1_OverallQual"] = data["BsmtFinSF1"] * data["OverallQual"]
df["-_LotArea_OverallQual"] = data["LotArea"] * data["OverallQual"]
df["-_TotalHouse_LotArea"] = df["TotalHouse"] + data["LotArea"]
df["Bsmt"] = data["BsmtFinSF1"] + data["BsmtFinSF2"] + data["BsmtUnfSF"]
df["Rooms"] = data["FullBath"] + data["TotRmsAbvGrd"]
df["PorchArea"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]
df["TotalPlace"] = df["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"] + data["GarageArea"] + data["OpenPorchSF"] + \
                   data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]

# 将所有的类别型特征，one-hot编码
df = pd.get_dummies(df)
# 查看数据维度
df.shape

# 构建模型
# 导入GBDT算法
from sklearn.ensemble import GradientBoostingRegressor
# 导入均方误差计算
from sklearn.metrics import mean_squared_error
# 导入标准化模块包
from sklearn.preprocessing import RobustScaler
# 导入划分数据集包，交叉验证包
from sklearn.model_selection import train_test_split, KFold, cross_val_score
# 导入Xgboost算法包
import xgboost as xgb

# 特征标准化
x = RobustScaler().fit_transform(df.drop(['SalePrice'], axis=1).values)
# 提取标签
y = df['SalePrice'].values


# 定义验证函数,使用5折交叉验证，采用均方根误差判别，返回均方根误差
def rmse_cv(model):
    # 将数据集shuffle打乱，划分成五分
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    # 计算均方根误差，其输出结果有五个
    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse


# GBDT算法
# 使用GBDT算法，构建模型
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.005,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state=5)
score1 = rmse_cv(GBoost)
# 输出五个均方根误差的平均值RSME和其标准差SD,保留4位小数
print("Gradient Boosting score: RSME={:.4f} (SD={:.4f})\n".format(score1.mean(), score1.std()))

# Xgboost算法,构建模型
Xgboost = xgb.XGBRegressor(colsample_bytree=0.36, gamma=0.042,
                           learning_rate=0.05, max_depth=3,
                           min_child_weight=1.88, n_estimators=2200,
                           reg_alpha=0.4640, reg_lambda=0.8571,
                           subsample=0.5213, silent=1,
                           random_state=1, nthread=-1)
score2 = rmse_cv(Xgboost)
# 输出五个均方根误差的平均值RSME和其标准差SD,保留4位小数
print("Xgboost score: RSME={:.4f} (SD={:.4f})\n".format(score2.mean(), score2.std()))

# 将80%数据作为训练集，20%数据作为测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# 定义回归拟合图,以预测值输入
def drawing(y_hat):
    # 获取预测的测试集从小到大排序的索引
    order = np.argsort(y_hat)
    # 将测试集和预测的测试集按索引排序
    y_test_new = y_test[order]
    y_hat = y_hat[order]
    # 画图展示
    plt.figure(figsize=(8, 6), facecolor='w')
    t = np.arange(len(y_test))
    plt.plot(t, y_test_new, 'b-', linewidth=2, label='True')
    plt.plot(t, y_hat, 'r-', linewidth=2, label='Predicted')
    plt.legend(loc='upper left')
    plt.grid(b=True)
    plt.show()


# 使用GBDT算法，构建模型
# 训练集训练
GBoost.fit(x_train, y_train)
# 测试集结果预测
y_hat1 = GBoost.predict(x_test)
# 分别输出均方根误差RMSE，训练集和测试集的拟合优度R2
print("RMSE =  %.4f" % np.sqrt(np.mean((y_hat1 - y_test) ** 2)))
print('R2_train = %.4f' % GBoost.score(x_train, y_train))
print('R2_test = %.4f' % GBoost.score(x_test, y_test))
# 画出拟合效果图，蓝色表示真实值，红色为预测值
drawing(y_hat1)

# 使用Xgboost算法，构建模型
# 训练集训练
Xgboost.fit(x_train, y_train)
# 测试集结果预测
y_hat2 = Xgboost.predict(x_test)
# 分别输出均方根误差RMSE，训练集和测试集的拟合优度R2
print("RMSE =  %.4f" % np.sqrt(np.mean((y_hat2 - y_test) ** 2)))
print('R2_train = %.4f' % GBoost.score(x_train, y_train))
print('R2_test = %.4f' % GBoost.score(x_test, y_test))
# 画出拟合效果图 ，蓝色表示真实值，红色为预测值
drawing(y_hat2)
