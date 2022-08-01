# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:58:03 2022

@author: xyuser
"""

import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#正常显示画图时出现的中文和负号
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


BATCH_SIZE = 16                                 # 每一小批抽样样本数量
LR = 0.001                                       # 学习率
#EPSILON = 0.9                                   # greedy policy，我们用指数衰减形式的，所以在dqn属性中定义
GAMMA = 0.5                                     # 择时策略应该更关注短期收益
TARGET_REPLACE_ITER = 50                        # 目标网络更新频率
MEMORY_CAPACITY = 32                            # 记忆库容量

horizon =  5                                   # 预测区间
lookback = 5                                    # 回看区间
N_ACTIONS = 3                                   # 三个动作：做多，做空，持仓
N_STATES = 4*lookback
TC =  5/10000                                   # 交易手续费：万分之五

# 定义Net类;参考cartpole.py，网络部分按照研报做了改动
class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 128)                                         # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到128个神经元
        self.fc1.weight.data.normal_(0, 1)                                      # 权重初始化 (均值为0，方差为.1的正态分布)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)                                          # 设置第一个全连接层(输入层到隐藏层): 128个神经元到256个神经元
        self.fc2.weight.data.normal_(0, 1)                                      # 权重初始化 (均值为0，方差为.1的正态分布)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, N_ACTIONS)                                    # 设置第三个全连接层(隐藏层到输出层): 256个神经元到动作数个神经元
        self.fc3.weight.data.normal_(0, 1)                                      # 权重初始化 (均值为0，方差为.1的正态分布)
        self.sm = nn.Softmax(dim=1)
        
    def train_forward(self, x):                                                       # 定义forward函数 (x为状态)
        self.bn1.train()
        self.bn2.train()
        x = (self.fc1(x))                                                       # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x = self.bn1(x)
        x = F.relu(x)
        x = (self.fc2(x))
        x = self.bn2(x)
        x = F.relu(x)
        actions_value = self.sm(self.fc3(x))                                    # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值
    
    def eval_forward(self, x):
        self.bn1.eval()
        self.bn2.eval()
        x = (self.fc1(x))                                                       # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        x = self.bn1(x)
        x = F.relu(x)
        x = (self.fc2(x))
        x = self.bn2(x)
        x = F.relu(x)
        actions_value = self.sm(self.fc3(x))                                    # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value  

# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.SmoothL1Loss()                                      # 使用L1Loss
        self.EPSILON = 0.05+(0.9-0.05)*np.exp(-self.memory_counter/500)         # epsilon随着训练过程逐渐减小，即实现从探索到利用的过程

    def choose_action(self, x):
        x = torch.FloatTensor(x)
        x = x.contiguous().view(1,N_STATES)                                                                           
        #x = torch.unsqueeze(x, 0)                                               # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() > self.EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果大于EPSILON，选择最优动作
            actions_value = self.eval_net.eval_forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0,1,2
        return action                                                           # 返回选择的动作 

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        s = s.reshape(20)
        s_ = s_.reshape(20)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        #if self.learn_step_counter % TARGET_REPLACE_ITER == 0:                  # 一开始触发，然后每5步触发
            #self.target_net.load_state_dict(self.eval_net.state_dict())         # 将评估网络的参数赋给目标网络
        #self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)            # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行20列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行20列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net.train_forward(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net.train_forward(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用L1损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数


df_train = pd.read_csv('train.csv')                                     # 数据准备
df_train.drop(columns=['trade_date.1'],inplace=True)
#df_train['trade_date'] =  pd.to_datetime(df_train['trade_date'])

class stock_market():
    '''股票市场的模拟环境'''
    def __init__(self,df):
        self.lookback = lookback
        self.data = df[['Z_close','Z_open','Z_high','Z_low','trade_date','close']]
        self.curindex = 4
        # 从第五天收盘时开始，因为需要五天数据作为DQN输入
        self.state = {}
        self.pos = 0                                             # 当前持仓：0表示空仓，1表示持仓，不允许做空，但为了方便训练奖励中允许做空
        
    def preprocess(self):
        # 训练前，把每天收盘后更新的状态存在字典里
        for index in self.data.index[self.lookback-1:]:
            # 从第五天收盘时开始
            temp = np.array(self.data.iloc[index-self.lookback+1:index+1,:-2])
            # 某天收盘时状态
            self.state[self.data['trade_date'][index]] = temp
    
    def step(self,action):
        # 获取预测区间后状态和奖励,动作设置：0为做空，1为保持，2为做多
        # self.curindex += horizon 
        return self.state[self.data['trade_date'][self.curindex + horizon]],self.calc_reward(action)
    
    def calc_reward(self,action):
        # 根据action，当前的日期（index）和当前的持仓情况（pos）计算reward
        if self.pos == 0:
            if action == 0 or action == 1:
                return 100*(1-self.data['close'][self.curindex+horizon]/self.data['close'][self.curindex])
            else:
                return 100*((1-TC)*(self.data['close'][self.curindex+horizon]/self.data['close'][self.curindex])-1)
        
        else:
            if action == 0:
                return 100*((1-TC)*(2-self.data['close'][self.curindex+horizon]/self.data['close'][self.curindex])-1)
            else:
                return 100*(self.data['close'][self.curindex+horizon]/self.data['close'][self.curindex]-1)
        

episodes = 30
dqn = DQN()
# 加载之前训练到一半的策略（可选）
#dqn.eval_net = torch.load('trained_dqn.pt')
#dqn.target_net = torch.load('trained_dqn.pt')


### 开始训练

for i in range(episodes):
    print('<<<<<<<<<Episode: %s' % str(i+1))
    env = stock_market(df_train)
    env.preprocess()
    if i % 3 == 0:
        dqn.target_net.load_state_dict(dqn.eval_net.state_dict())
    while env.curindex < len(env.data)-horizon:
        s = env.state[env.data['trade_date'][env.curindex]]
        a = dqn.choose_action(s)
        s_ , r = env.step(a)
        if a == 0 and env.pos == 1:
            env.pos = 0
        elif a == 2 and env.pos == 0:
            env.pos = 1
        dqn.store_transition(s, a, r, s_)
        # s = s_
        if dqn.memory_counter > MEMORY_CAPACITY:             
            dqn.learn()
        env.curindex += 1
        # 遍历每一天进行训练

torch.save(dqn.target_net, 'trained_dqn.pt')
# 保存训练好的网络



### 开始测试

df_test = pd.read_csv('test.csv')
df_test.drop(columns=['trade_date.1'],inplace=True)
df_test['signal'] = np.ones(len(df_test))
df_test['reward'] = np.zeros(len(df_test))


# 用训练好的模型在测试集上回测，
# 策略描述：一旦DQN网络发出买入/卖出的信号，必须持有/平仓五天（即五天内保持仓位不变）

env = stock_market(df_test)
env.preprocess()
dqn.EPSILON = 0

# 初始化状态：第五天收盘时
    
while env.curindex < len(env.data) - horizon:
    s = env.state[env.data['trade_date'][env.curindex]]
    a = dqn.choose_action(s)
    s_,r = env.step(a)
    df_test.loc[env.curindex,'signal'] = a
    df_test.loc[env.curindex,'reward'] = r
    # 这里的reward不是真实收益，只起参考作用
    env.curindex += 1

# 保存结果，在notebook里可视化与进一步分析
df_test.to_csv('res.csv')

# 作图

# 根据每天换仓的动作计算持仓情况
df_test['position'] = df_test['signal'].shift(1)
df_test['position'] = df_test['position'].replace({1:np.nan})
df_test['position'] = df_test['position'].replace({2:1})
df_test['position'].fillna(method='ffill',inplace = True)
df_test['position'].fillna(0,inplace=True)
df_test['ret'] = df_test['close'].pct_change()
df_test.loc[0,'ret']=0

cost = 5/10000
#根据交易信号和仓位计算策略的每日收益率
df_test.loc[df_test.index[0], 'capital_ret'] = 0
#今天开盘新买入的position在今天的涨幅(扣除手续费)
df_test.loc[df_test['position'] > df_test['position'].shift(1), 'capital_ret'] = \
                         (df_test['close'] / df_test['open']-1) * (1- cost) 
#卖出同理
df_test.loc[df_test['position'] < df_test['position'].shift(1), 'capital_ret'] = \
                   (df_test['open'] / df_test['close'].shift(1)-1) * (1-cost) 
# 当仓位不变时,当天的capital是当天的change * position
df_test.loc[df_test['position'] == df_test['position'].shift(1), 'capital_ret'] = \
                        df_test['ret'] * df_test['position']

df_test['capital_line']=(df_test.capital_ret+1.0).cumprod()
df_test['index_line']=(df_test.ret+1.0).cumprod()

df_test.set_index(pd.to_datetime(df_test['trade_date']),inplace=True)

plt.figure(figsize=(20,8))
plt.plot(df_test['capital_line'],label='强化学习择时策略净值')
plt.plot(df_test['index_line'],label='上证指数净值')
plt.legend()
plt.show()
