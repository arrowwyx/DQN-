## This project presents a simple trading strategy based on DQN

get_data那个notebook如果想跑要用自己的token


接下来大概的方向：改进所用因子，用期货数据（回测时允许做多和做空，现在回测时只能做多），各种调参和神经网络的创新（？）

update 2022.8.1: 加入BN层，现在能稳定跑出尚可的超额收益

update 2022.8.10: 尝试了MACD，RSI等技术类因子和Alpha101中一些容易实现的因子做组合，收益没有明显提升
