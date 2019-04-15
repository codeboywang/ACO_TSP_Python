import numpy as np
import random

class TSP_ACO (object):
    def __init__(self):
        self.Maximum  =  np.inf    # 设定最大值
        self.readCitMapfromTxt()
        self.citymap = np.array(np.full((self.mapSize,self.mapSize),self.Maximum)) # citymap[i][j] 表示 城市i，j的距离
        self.visionDistsnce = np.array(np.full((self.mapSize, self.mapSize),0))  # 能见度 是 距离的倒数
        # 运算self.citymap
        for i in range(len(self.citycoordinates)):
            for j in range(i,len(self.citycoordinates)):
                if i==j :
                    self.citymap[i][j] = 0
                else :
                    # self.citymap[i][j] 记录 city i 与 city j 的笛卡尔距离
                    self.citymap[i][j] = np.sqrt(np.power((self.citycoordinates[i]['x'] - self.citycoordinates[j]['x']),2) +
                                            np.power((self.citycoordinates[i]['y'] - self.citycoordinates[j]['y']), 2))
                    self.citymap[j][i] = self.citymap[i][j]
                    self.visionDistsnce[i][j] = 1 / self.citymap[i][j]
                    self.visionDistsnce[j][i] = 1 / self.citymap[j][i]

        self.antNumber                  = 50    # 蚁群规模
        self.initialInfomationRemain    = 100    # 初始化的信息素
        self.informationMap             = np.array(np.full((self.mapSize, self.mapSize), self.initialInfomationRemain))
        self.alpha                      = 1     # a值，a越大，说明该蚂蚁更倾向于走其他蚂蚁走过的路径
        self.beta                       = 5    # b值，b越大，说明该蚂蚁更倾向于局部信息作出判断，达成局部最优解
        self.rho                        = 0.9   # Rho值，信息素残留常数
        self.heuristicRate              = 1    # 启发式的比例
        self.infomationFadeOutRate      = 1 - self.rho  # 信息素挥发率，越大，之前搜索过的路径再被搜索的概率也大，
                                                        # 越小，提高随机性与全局搜索能力，但是算法收敛程度降低

        self.Q                          = 100   # Q值，为一圈下来，一只蚂蚁能释放出的信息素的数量
        self.iterateNumber              = 30    # 设计迭代次数
        self.antLoopDistance            = [0 for i in range(self.antNumber)] # 蚂蚁跑完一圈后的路程
        self.minDistance                = 1e9   # 最短路径的长度
        self.minDistanceTrace           = []    # 存最短路径的序列
        self.ant                        = []    # 记录蚂蚁的轨迹

    # 初始化蚂蚁的位置
    def setAnt(self):
        for i in range(self.antNumber):
            startCity = random.randint(0,self.mapSize-1)
            self.ant.append([startCity]) # 闭区间

    # 蚁群算法主要函数
    def doACO(self):
        self.setAnt()
        for l in range(self.iterateNumber):
            print("In "+str(l)+" iteration!")
            # 蚂蚁i开始依次搜索
            for i in range(self.antNumber):
                # 蚂蚁i没有跑圈就继续搜索
                while len(self.ant[i]) < self.mapSize:
                    # 蚂蚁根据信息素和启发式选择下一个城市
                    choiceCity = self.antChoice(i)
                    self.antLoopDistance[i] += self.citymap[self.ant[i][-1]][choiceCity]
                    self.ant[i].append(choiceCity)

                # 蚂蚁i已经走到了一圈
                self.antLoopDistance[i] += self.citymap[self.ant[i][0]][self.ant[i][-1]]
                if self.antLoopDistance[i] < self.minDistance :
                    self.minDistance = self.antLoopDistance[i]
                    print("min ant " + str(i)+ " trace as :" ,self.ant[i])
                    self.minDistanceTrace = self.ant[i].copy()

            # 所有蚂蚁搜索完了，信息素褪去
            self.informationFadeOut()

            # 蚂蚁搜索完了要留下，新的信息素
            self.updateInformationMap()

            # 更新重要信息
            self.updateAntInformation()

        print("Finish search!")
        print("Min distance is :", self.minDistance)
        # print("Trace is :",self.minDistanceTrace)
        print("Trace is :")
        for i in self.minDistanceTrace:
            print(self.cityNumber[i], end=' ')


    # 信息素褪去
    def informationFadeOut(self):
        for i in range(len(self.informationMap)):
            for j in range(len(self.informationMap[i])):
                self.informationMap[i][j] *= self.rho

    # 更新信息素
    def updateInformationMap(self):
        # 计算第i只蚂蚁更新的信息素
        for i in range(self.antNumber):
            informationAllocation = np.array(np.full((self.mapSize, self.mapSize), float(0)))
            for j in range(len(self.ant[i])):
                # 起始城市与末位城市相连
                if j == 0:
                    self.informationMap[self.ant[i][j]][self.ant[i][-1]] += self.Q * self.citymap[self.ant[i][j]][self.ant[i][-1]] / self.antLoopDistance[i]
                    self.informationMap[self.ant[i][-1]][self.ant[i][j]] += self.Q * self.citymap[self.ant[i][-1]][self.ant[i][j]] / self.antLoopDistance[i]
                else:
                    self.informationMap[self.ant[i][j]][self.ant[i][j-1]] += self.Q * self.citymap[self.ant[i][j]][self.ant[i][j-1]] / self.antLoopDistance[i]
                    self.informationMap[self.ant[i][j-1]][self.ant[i][j]] += self.Q * self.citymap[self.ant[i][j-1]][self.ant[i][j]] / self.antLoopDistance[i]

    # 更新重要信息
    def updateAntInformation(self):
        for i in range(self.antNumber):
            startCity = self.ant[i][0]
            del self.ant[i][:]
            self.ant[i].append(startCity)
            self.antLoopDistance[i] = 0
            # print(self.ant[i])

    # 蚂蚁做路径选择
    def antChoice(self,antNumber):
        curruntCity  = self.ant[antNumber][-1]
        visitedCity  = set(self.ant[antNumber])
        allowedCity  = set(range(0,self.mapSize)) - visitedCity
        # 计算启发式参数，距离越近概率越大
        distanceSum  = float(0)
        for city in allowedCity:
            distanceSum += self.citymap[curruntCity][city]

        # 启发式初始为0
        heuristic = np.array(np.full((self.mapSize), float(0)))
        for city in allowedCity:
            heuristic[city] = self.heuristicRate * distanceSum / self.citymap[curruntCity][city]

        # 计算该蚂蚁所有可能性
        posibility      = [float(0) for i in range(self.mapSize)]
        posibilitySum   = float(0)
        for city in allowedCity:
            posibility[city] = np.power(self.informationMap[curruntCity][city],self.alpha) * np.power(heuristic[city],self.beta)
            posibilitySum += posibility[city]

        # 计算最后可能性
        finalPosibility = [float(0) for i in range(self.mapSize)]
        for city in allowedCity:
            finalPosibility[city] = posibility[city] / posibilitySum

        # 最后根据可能性中随机选择
        # 此处存在问题
        anchor = random.random()
        psum   = float(0)
        for choiceCity in range(self.mapSize) :
            if psum < anchor and ( psum + finalPosibility[choiceCity] ) > anchor :
                return  choiceCity
            psum += finalPosibility[choiceCity]

        return allowedCity.pop()



    # 从文本中读取城市数据
    def readCitMapfromTxt(self):
        # 读取城市坐标文件
        f   = open("citymap")
        txt = f.read()
        f.close()
        # 运算self.citycoordinates
        items = txt.split("\n")
        self.mapSize = len(items)
        self.citycoordinates = []
        self.cityNumber      = [] # 一个算法城市标号与输入城市标号的映射，cityNumber[i] = j 表示 在算法中i号城市，是输入的j号城市
                                  # 算法中城市标号从0开始
        for item in items:
            triple = item.split(" ")
            self.cityNumber.append(int(triple[0]))
            self.citycoordinates.append({"x":int(triple[1]),"y":int(triple[2])})


obj_ACO = TSP_ACO()

obj_ACO.doACO()