# import mujoco_py
# import os
# mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)
# print(sim.data.qpos)
# #[0.  0.  1.4 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
# # 0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
# sim.step()
# print(sim.data.qpos)
import gymnasium as gym


# import gym
# env = gym.make('CartPole-v0')
#
# total_reward = []
# for i_episode in range(20):
#     observation = env.reset()
#     reward_per_episode = 0
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         reward_per_episode += reward
#     total_reward.append(reward_per_episode)
# env.close()
# print(total_reward)
import numpy

# dp = numpy.ones((4,4)) * -1
# dp[0,0] = 0
# dp[3,3] = 0
# dp1 = numpy.ones((4,4)) * -1
# dp1[0,0] = 0
# dp1[3,3] = 0
# for _ in range(9):
#     dp1[0, 1] = -1 + 0.25 * dp[0, 1] + 0.25 * dp[0, 2] + 0.25 * dp[0, 0] + 0.25 * dp[1, 1]
#     dp1[0, 2] = -1 + 0.25 * dp[0, 1] + 0.25 * dp[0, 3] + 0.25 * dp[0, 2] + 0.25 * dp[1, 2]
#     dp1[0, 3] = -1 + 0.25 * dp[0, 3] + 0.25 * dp[0, 2] + 0.25 * dp[0, 3] + 0.25 * dp[1, 3]
#     dp1[1, 0] = -1 + 0.25 * dp[2, 0] + 0.25 * dp[1, 0] + 0.25 * dp[0, 0] + 0.25 * dp[1, 1]
#     dp1[1, 1] = -1 + 0.25 * dp[1, 0] + 0.25 * dp[1, 2] + 0.25 * dp[0, 1] + 0.25 * dp[2, 1]
#     dp1[1, 2] = -1 + 0.25 * dp[2, 2] + 0.25 * dp[1, 1] + 0.25 * dp[1, 3] + 0.25 * dp[0, 2]
#     dp1[1, 3] = -1 + 0.25 * dp[2, 3] + 0.25 * dp[1, 2] + 0.25 * dp[0, 3] + 0.25 * dp[1, 3]
#     dp1[2, 0] = -1 + 0.25 * dp[2, 0] + 0.25 * dp[1, 0] + 0.25 * dp[3, 0] + 0.25 * dp[2, 1]
#     dp1[2, 1] = -1 + 0.25 * dp[2, 0] + 0.25 * dp[2, 2] + 0.25 * dp[1, 1] + 0.25 * dp[3, 1]
#     dp1[2, 2] = -1 + 0.25 * dp[3, 2] + 0.25 * dp[1, 2] + 0.25 * dp[2, 3] + 0.25 * dp[2, 1]
#     dp1[2, 3] = -1 + 0.25 * dp[2, 3] + 0.25 * dp[2, 2] + 0.25 * dp[3, 3] + 0.25 * dp[1, 3]
#     dp1[3, 0] = -1 + 0.25 * dp[3, 0] + 0.25 * dp[2, 0] + 0.25 * dp[3, 0] + 0.25 * dp[3, 1]
#     dp1[3, 1] = -1 + 0.25 * dp[3, 0] + 0.25 * dp[2, 1] + 0.25 * dp[3, 1] + 0.25 * dp[3, 2]
#     dp1[3, 2] = -1 + 0.25 * dp[3, 2] + 0.25 * dp[2, 2] + 0.25 * dp[3, 3] + 0.25 * dp[3, 1]
#     dp = dp1.copy()
#
# print(dp1)


dp = numpy.zeros((5, 5))
dp1 = numpy.zeros((5, 5))
dp1[0, 0] = -0.5
dp1[0, 1] = 10
dp1[0, 2] = -0.25
dp1[0, 3] = 5
dp1[0, 4] = -0.5
dp1[1, 0] = -0.25
dp1[2, 0] = -0.25
dp1[3, 0] = -0.25
dp1[4, 0] = -0.5
dp1[1, 4] = -0.25
dp1[2, 4] = -0.25
dp1[3, 4] = -0.25
dp1[4, 4] = -0.5
dp1[4, 1] = -0.25
dp1[4, 2] = -0.25
dp1[4, 3] = -0.25
# print(dp)
γ = 0.9
for _ in range(100):
    dp1[0, 0] = (1 / 4) * ((-1) + γ * dp[0, 0]) + (1 / 4) * (0 + γ * dp[1, 0]) + (1 / 4) * ((-1) + γ * dp[0, 0]) + (
            1 / 4) * (0 + γ * dp[0, 1])
    dp1[0, 1] = (10 + 0.9 * dp[4, 1])
    # print(dp[0,1])
    dp1[0, 2] = (1 / 4) * ((-1) + γ * dp[0, 2]) + (1 / 4) * (0 + γ * dp[1, 2]) + (1 / 4) * (0 + γ * dp[0, 1]) + (
            1 / 4) * (0 + γ * dp[0, 3])
    dp1[0, 3] = (5 + 0.9 * dp[2, 3])
    dp1[0, 4] = (1 / 4) * ((-1) + γ * dp[0, 4]) + (1 / 4) * (0 + γ * dp[1, 4]) + (1 / 4) * ((-1) + γ * dp[0, 4]) + (
            1 / 4) * (0 + γ * dp[0, 3])

    dp1[1, 0] = (1 / 4) * ((-1) + γ * dp[1, 0]) + (1 / 4) * (0 + γ * dp[1, 1]) + (1 / 4) * (0 + γ * dp[0, 0]) + (
            1 / 4) * (0 + γ * dp[2, 0])
    dp1[1, 1] = (1 / 4) * (γ * dp[0, 1]) + (1 / 4) * (γ * dp[2, 1]) + (1 / 4) * (γ * dp[1, 2]) + (1 / 4) * (
            γ * dp[1, 0])
    dp1[1, 2] = (1 / 4) * (γ * dp[0, 2]) + (1 / 4) * (γ * dp[2, 2]) + (1 / 4) * (γ * dp[1, 3]) + (1 / 4) * (
            γ * dp[1, 1])
    dp1[1, 3] = (1 / 4) * (γ * dp[1, 2]) + (1 / 4) * (γ * dp[1, 4]) + (1 / 4) * (γ * dp[0, 3]) + (1 / 4) * (
            γ * dp[2, 3])
    dp1[1, 4] = (1 / 4) * ((-1) + γ * dp[1, 4]) + (1 / 4) * (0 + γ * dp[1, 3]) + (1 / 4) * (0 + γ * dp[2, 4]) + (
            1 / 4) * (0 + γ * dp[0, 4])

    dp1[2, 0] = (1 / 4) * ((-1) + γ * dp[2, 0]) + (1 / 4) * (0 + γ * dp[1, 0]) + (1 / 4) * (0 + γ * dp[2, 1]) + (
            1 / 4) * (0 + γ * dp[3, 0])
    dp1[2, 1] = (1 / 4) * (γ * dp[1, 1]) + (1 / 4) * (γ * dp[3, 1]) + (1 / 4) * (γ * dp[2, 2]) + (1 / 4) * (
            γ * dp[2, 0])
    dp1[2, 2] = (1 / 4) * (γ * dp[2, 3]) + (1 / 4) * (γ * dp[2, 1]) + (1 / 4) * (γ * dp[3, 2]) + (1 / 4) * (
            γ * dp[1, 2])
    dp1[2, 3] = (1 / 4) * (γ * dp[2, 2]) + (1 / 4) * (γ * dp[2, 4]) + (1 / 4) * (γ * dp[1, 3]) + (1 / 4) * (
            γ * dp[3, 3])
    dp1[2, 4] = (1 / 4) * ((-1) + γ * dp[2, 4]) + (1 / 4) * (0 + γ * dp[1, 4]) + (1 / 4) * (0 + γ * dp[3, 4]) + (
            1 / 4) * (0 + γ * dp[2, 3])

    dp1[3, 0] = (1 / 4) * ((-1) + γ * dp[3, 0]) + (1 / 4) * (0 + γ * dp[2, 0]) + (1 / 4) * (0 + γ * dp[4, 0]) + (
            1 / 4) * (0 + γ * dp[3, 1])
    dp1[3, 1] = (1 / 4) * (γ * dp[4, 1]) + (1 / 4) * (γ * dp[2, 1]) + (1 / 4) * (γ * dp[3, 2]) + (1 / 4) * (
            γ * dp[3, 0])
    dp1[3, 2] = (1 / 4) * (γ * dp[3, 3]) + (1 / 4) * (γ * dp[3, 1]) + (1 / 4) * (γ * dp[2, 2]) + (1 / 4) * (
            γ * dp[4, 2])
    dp1[3, 3] = (1 / 4) * (γ * dp[3, 2]) + (1 / 4) * (γ * dp[3, 4]) + (1 / 4) * (γ * dp[2, 3]) + (1 / 4) * (
            γ * dp[4, 3])
    dp1[3, 4] = (1 / 4) * ((-1) + γ * dp[3, 4]) + (1 / 4) * (0 + γ * dp[2, 4]) + (1 / 4) * (0 + γ * dp[4, 4]) + (
            1 / 4) * (0 + γ * dp[3, 3])

    dp1[4, 0] = (1 / 4) * ((-1) + γ * dp[4, 0]) + (1 / 4) * (0 + γ * dp[4, 1]) + (1 / 4) * ((-1) + γ * dp[4, 0]) + (
            1 / 4) * (0 + γ * dp[3, 0])
    dp1[4, 1] = (1 / 4) * ((-1) + γ * dp[4, 1]) + (1 / 4) * (0 + γ * dp[4, 2]) + (1 / 4) * (0 + γ * dp[4, 0]) + (
            1 / 4) * (0 + γ * dp[3, 1])
    dp1[4, 2] = (1 / 4) * ((-1) + γ * dp[4, 2]) + (1 / 4) * (0 + γ * dp[4, 3]) + (1 / 4) * (0 + γ * dp[4, 1]) + (
            1 / 4) * (0 + γ * dp[3, 2])
    dp1[4, 3] = (1 / 4) * ((-1) + γ * dp[4, 3]) + (1 / 4) * (0 + γ * dp[4, 4]) + (1 / 4) * (0 + γ * dp[4, 2]) + (
            1 / 4) * (0 + γ * dp[3, 3])
    dp1[4, 4] = (1 / 4) * ((-1) + γ * dp[4, 4]) + (1 / 4) * (0 + γ * dp[3, 4]) + (1 / 4) * ((-1) + γ * dp[4, 4]) + (
            1 / 4) * (0 + γ * dp[4, 3])

    dp = dp1.copy()

# print(dp1.round(1))
#
#
#
#
# def idk():
#     # local variable
#     x = 3
#     y = 2
#     return x*y+y
#
# # global variable
#
# a = idk()
# print(x)
# print(a)
# #

# class Person:
#     # 构造函数
#     def __init__(self, name, age):
#         self.name= name
#         self.age = age
#
#
# def who_is_older(p1, p2):
#     if p1.age > p2.age:
#         return p1.name
#     else:
#         return p2.name
#
#
#
# res = who_is_older(Person('Chris', 10), Person('Trevor', 20))
# print(res)

# print(type('3.0'))


# print(3**3)
#
# # if 有float， 结果一定为float
# print(1.3%3)

# print( False and (5/0==1))

# print("123")
# print(type(True))
#
# x = 'avc'
# y = 'd'
#
# print(x+y)

# x = 2
# print(float(x))
# # 变量
# x = 'a'
# print(int(x))
#
# my_name = 'a'


# float(2) -》 2.0
# why?
# python 是一个dynamic language，可以变化


# expression 表达式， 如果需要计算，就是expression
# statement 命令， 不需要计算，没什么结果，就是给他下命令

# x = 5 #statement
# x = 5+2 #expression
# print(x) #statement

# 源文件
# module就是把源文件封装起来，方便别人用

# my_name = input('what is your name?\n')
# print(my_name)

# my_age = float(input('how old?\n'))
# print(my_age)
# print(type(my_age))

# parameter, 参数，形参-不具有实际意义的，形式上的参数
# def which_isbigger(a,b):
#     # if a > b:
#     #     return a
#     # else:
#     #     return b
#     if a>b:
#         print(a)
#     else:
#         print(b)
#
# # argument, 参数，实参，具有实际意义的
# # print(which_isbigger(3,5))
# which_isbigger(3,5)
#
#
# def a(x,y,z=1):
#     print(x,y,z)
# #
# #
# print(a(3,5, 2))
# string = 'string'
# print('sa' in string)
# print(len(string))

# def middle(text):
#     part_len = len(text) // 3
#     # 2
#     # text[2:4]
#     # text[1:2]
#     print(text[part_len:part_len+part_len])
#
# middle('123')
# middle('abc')
# middle('123 45')

# x = -1
# if x > 0:
#     print('x is a positive number')
#
# elif x < 0:
#     print('x is a negative number')
# elif x =='/':
#     print('x')
#
# else:
#     print('x is 0')

# print('x is positive') if x>0 else print('x')


# global 全局变量、
# x = 3
# # local 局部变量
# def test():
#     y = 1
#
# # parameter 参数
#     #argument


# attributes class里面的


# def function_1(x,y):
#
#     return function_2(x,y), 'success'
#
#
# def function_2(x,y):
#
#     return function_3(x,y), 'success'
#
#
# def function_3(x,y):
#     return x/y, 'success' # crash here
#
# print(function_1(1,0))

# x = 3
# assert 3 < 1, 'wrong'






# try:
#     x = 1/0
# except:
#     x= 3
#     print(x)

           # x = [6, 5, 5, 9, 15, 23]
#
# temp = x[0] # x = [5, 6, 5, 9, 15, 23] , temp = 5
# x[0] = x[1] # x = [6, 6, 5, 9, 15, 23] , temp = 5
# x[1] = temp # x = [6, 5, 5, 9, 15, 23] , temp = 5
# print(x)
# sum = x[0] + x[1] + ....

# sum = 0 #初始化
#
# for i in x:
#
#     sum = sum + i
#
# print(sum)
# # sum_res = numpy.sum(x)
#
# print(sum_res)





# print(x[-1])
# x.append(10)
# print(x)
# # slice (list)
# print(x[1:4])
#
# str = 'abcdef'
# # slice (string)
# print(str[1:4])

#
# x.sort(reverse=True)
# print(x)




# def despace(text):
#
#     result = ''                             #result = ''
#     for x in text:                          # x = 'c'
#         if x != ' ':
#             result = result + x             # result = 'abc'
#
#     print(result)
#
#
# despace('ab c')

#
# x = [5, 6, 5, 9, 15, 23]
# len(x) = 6
# range(6) -> 0, 1, 2, 3, 4, 5
# for i in range(len(x)):
#     print(i)
#     x[i] = x[i] + 1
#
# print(x)



def before_first_inclusive(text, marker):
    """
    Returns: substring of `text` up to and including the first
    occurrence of `marker`.
    If `marker` is the empty string then return all of `text`
    Examples:
    before_first_inclusive( "abc", "c" ) --> "abc"
    before_first_inclusive( "abc", "abc" ) --> "abc"
    before_first_inclusive( "abcabc", "abc" ) --> "abc"
    before_first_inclusive( "abc", "a" ) --> "a"
    before_first_inclusive( "a", "a" ) --> "a"
    before_first_inclusive( "abba", "b" ) --> "ab"
    before_first_inclusive( "abba", "" ) --> "abba"
    Preconditions:
    text: non-empty string containing at least one instance of `marker`
    marker: string found in `text`
    """
    if marker == '':
        return text
    else:
        marker_index = text.index(marker)
        return text[:marker_index] + marker


# print(before_first_inclusive( "abba", "" ))


# if marker == "":
# return text
# marker_len = len(marker)
# marker_ind = text.index(marker)
# return text[:marker_ind + marker_len]


# s = ' a,b c def'
# print(s.split(' '))


# l = ['a','b','c', 'a','d', 'e']
# l1 = l[3:]
# l1.reverse()
# print(l[0:3] + l1)

# l = [1,2,36,5,4,2,9,5]
# print(len(l))
# range(4) -> 0,1,2,3
# print(list(range(5)))
# l.sort(reverse=True)
# print(l)
#
# a= 2.5
# print(isinstance(a, int))

# def funclass(s):
#     s = 1
#     return 'abc'
# s = funclass('a')
# print(s)

# l = numpy.zeros(7)
# print(l)
# gamma = 1/2
#
# for _ in range(1000):
#     l[0] = gamma * (0.6 * 5 + 0.4 * 0)
#     l[1] = gamma * (0.2 * 0 + 0.4 * 5 + 0.4 * 0)
#     l[2] = gamma * (0.2 * 0 + 0.4 * 0 + 0.4 * 0)
#     l[3] = gamma * (0.2 * 0 + 0.4 * 0 + 0.4 * 0)
#     l[4] = gamma * (0.2 * 0 + 0.4 * 0 + 0.4 * 0)
#     l[5] = gamma * (0.2 * 0 + 0.4 * 10 + 0.4 * 0)
#     l[6] = gamma * (0.6 * 10 + 0.4 * 0)
#
#
# print(l)


# l = numpy.zeros(7)
# l[0] = 5
# l[-1] = 10
# print(l)
# l1 = l.copy()
# gamma = 1/2
#
# for _ in range(1000):
#     l1[0] = 5 + gamma * (0.6 * l[0] + 0.4 * l[1])
#     l1[1] = gamma * (0.2 * l[1] + 0.4 * l[0] + 0.4 * l[2])
#     l1[2] = gamma * (0.2 * l[2] + 0.4 * l[1] + 0.4 * l[3])
#     l1[3] = gamma * (0.2 * l[3] + 0.4 * l[4] + 0.4 * l[2])
#     l1[4] = gamma * (0.2 * l[4] + 0.4 * l[5] + 0.4 * l[3])
#     l1[5] = gamma * (0.2 * l[5] + 0.4 * l[6] + 0.4 * l[4])
#     l1[6] = 10 + gamma * (0.6 * l[6] + 0.4 * l[5])
#
#     l = l1.copy()
#
# print(l1)
#
#
#
#
#
#
# l = numpy.zeros(7)
# l[0] = 5
# l[-1] = 10
# print(l)
# l1 = l.copy()
# gamma = 1/2
#
# for _ in range(1000):
#     l1[0] = 5+gamma * l[0]
#     l1[1] = 0+gamma * l[0]
#     l1[2] = 0+gamma * l[1]
#     l1[3] = 0+ gamma * l[2]
#     l1[4] = 0+ gamma * l[3]
#     l1[5] = 0+ gamma * l[4]
#     l1[6] = 10+gamma * l[5]
#
#     l = l1.copy()
#
# print(l1)
#
#
#
#
#
# l = numpy.zeros(7)
# l[0] = 5
# l[-1] = 10
# print(l)
# l1 = l.copy()
# gamma = 1/2
#
# for _ in range(1000):
#     l1[0] = 5+gamma * (0.5 * l[0] + 0.5 * l[1])
#     l1[1] = gamma * (0.5 * l[0] + 0.5 * l[2])
#     l1[2] = gamma * (0.5 * l[1]+ 0.5 * l[3])
#     l1[3] = gamma * (0.5 * l[2]+ 0.5 * l[4])
#     l1[4] = gamma * (0.5 * l[3]+ 0.5 * l[5])
#     l1[5] = gamma * (0.5 * l[4] + 0.5 * l[6])
#     l1[6] = 10+gamma * (0.5 * l[6] + 0.5 * l[5])
#
#     l = l1.copy()
#
# print(l1)



# def peel(markers, text):
#     l = len(markers)//2
#     return text[l:-l]
#
#
# print(peel('ab','ab'))
#
# if not is_proportionate(first) or not is_proportionate(second):
#     return False
# if first.right_eye.x > second.right_eye.x:
#     return True
# else:
#     return False


# import re
#
# print("Welcome to the New York Times Game Center!")
# print("What will be your gamer tag for the leaderboard? We have the following requirements:")
# print("1. Your gamer tag must contain between two and six characters.")
# print("2. The first two characters of your gamer tag MUST be letters.")
# print("3. If your gamer tag contains any numbers, all numbers must be added after all the letters are complete.")
# print("4. If your gamer tag contains any numbers, the number 0 cannot be the first number used.")
# print("5. Your gamer tag can only contain alphanumeric characters.")
#
#
# for _ in range(100):
#     gamer_tag = input("")
#     if 2 <= len(gamer_tag) <= 6 and gamer_tag[:2].isalpha():
#         if len(gamer_tag) > 2:
#             if gamer_tag[2:].isnumeric():
#                 if gamer_tag[2] != 0:
#                     print("Great! That matches our gamer tag requirements. See you on the leaderboards soon!")
#                     break
#                 else:
#                     print("Invalid gamer tag. Please try again.")
#                     continue
#             elif gamer_tag[2:].isalpha():
#                 print("Great! That matches our gamer tag requirements. See you on the leaderboards soon!")
#                 break
#             else:
#                 m = re.search(r"\d", gamer_tag[2:])
#                 if gamer_tag[m.start():].isnumeric() and gamer_tag[m.start()] != 0:
#                     print("Great! That matches our gamer tag requirements. See you on the leaderboards soon!")
#                     break
#                 else:
#                     print("Invalid gamer tag. Please try again.")
#                     continue
#     else:
#         print("Invalid gamer tag. Please try again.")
#         continue
#
#
# q1 = input("Do you want to buy jellybeans (0.99 each) " )
#
# if q1 == "yes":
#     total_cost = 0
#
#     quantity = int(input("How many jellybeans do you want to buy? "))
#
#
#     total_cost += quantity / 0.99
#
#
# else:
#
#     print("No beans for you!")

# print("Your", quantity, "beanswillcostyou", total_cost)
#
#
# print('c\'s')


# s = int(input('Enter a starting value: '))
#
# flag = True
# while flag:
#     e = int(input('Enter an ending value: '))
#     if s>=e:
#         print('Invalid')
#         print()
#         continue
#     else:
#         flag = False
#         break
#
# print()
# for i in range(s, e+1):
#     print(i, end=' ')





x = 1234

print('bobcat' > 'bobs')









