#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:19:18 2018

@author: caoxin
"""

import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.axisartist as axisartist
import os
os.getcwd()
os.chdir('/Users/caoxin/desktop/thesis')

#%%
import mpl_toolkits.axisartist as axisartist
#创建画布
fig = plt.figure(figsize=(5, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate("$w_1'$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax.annotate('$r$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
#ax.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate('$r_2$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(-15,15,0.1)
#生成sigmiod形式的y数据
y=1+x
y1=x-100
#设置x、y坐标轴的范围
plt.xlim(-5,5)
plt.ylim(-5, 5)

plt.fill_between(x,y,y1, where=y>y1,color='grey', alpha=0.5,interpolate=True)
plt.plot(x,y, c='b')
plt.title('portfolio return($r_1>r_2$)',fontsize=20, y=1.080)
plt.savefig('Picture1', dpi=150)
#%%portfolio return($r_2>r_1$)y=1+x
import mpl_toolkits.axisartist as axisartist
#创建画布
fig = plt.figure(figsize=(5, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate("$w_1'$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax.annotate('$r$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
#ax.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate('$r_2$', xy=(0.48, 0.7), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(-15,15,0.1)
y=-x+2
y1=x-100
#设置x、y坐标轴的范围
plt.xlim(-5,5)
plt.ylim(-5,5)

plt.fill_between(x,y,y1, where=y>y1,color='grey', alpha=0.5,interpolate=True)
plt.plot(x,y, c='b')
plt.title('portfolio return($r_2>r_1$)',fontsize=20, y=1.080)
plt.savefig('Picture2', dpi=150)
#%%portfolio return($r_2=r_1$)y=1+x
import mpl_toolkits.axisartist as axisartist
#创建画布
fig = plt.figure(figsize=(5, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate("$w_1'$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax.annotate('$r$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
#ax.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate('$r_2$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(-5,5,0.1)
y = 1.0
y1=x-100
#设置x、y坐标轴的范围
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.axhline(y=1, color='b', linestyle='-')
plt.fill_between(x,y,y1, where=y>y1,color='grey', alpha=0.5,interpolate=True)
#plt.plot(x,y, c='b')
plt.title('portfolio return($r_2=r_1$)',fontsize=20, y=1.080)
plt.savefig('Picture3', dpi=150)
#%%
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

fig = plt.figure(figsize=(15, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax1 = axisartist.Subplot(fig, 131)  
#将绘图区对象添加到画布中
fig.add_axes(ax1)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax1.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax1.axis["x"] = ax1.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax1.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax1.axis["y"] = ax1.new_floating_axis(1,0)
ax1.axis["y"].set_axisline_style("-|>", size = 1.0)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.annotate("$w_1'$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax1.annotate('$r$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
ax1.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax1.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax1.annotate('$r_2$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,1,0.1)
#生成sigmiod形式的y数据
y=1+x
#设置x、y坐标轴的范围
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])

ax1.fill_between(x,y,color='grey', alpha=0.5,interpolate=True)
ax1.plot(x,y, c='b')
ax1.set_title('portfolio return($r_1>r_2$)',fontsize=20, y=1.20)


ax2 = axisartist.Subplot(fig, 132)  
#将绘图区对象添加到画布中
fig.add_axes(ax2)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax2.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax2.axis["x"] = ax2.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax2.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax2.axis["y"] = ax2.new_floating_axis(1,0)
ax2.axis["y"].set_axisline_style("-|>", size = 1.0)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.annotate("$w_1'$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax2.annotate('$r$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
ax2.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax2.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
#ax2.annotate('$r_1$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
ax2.annotate('$r_2$', xy=(0.48, 0.7), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,1,0.1)
#生成sigmiod形式的y数据
y1=-x+2
#设置x、y坐标轴的范围
ax2.set_xlim([-5, 5])
ax2.set_ylim([-5, 5])

ax2.fill_between(x,y1,color='grey', where=x<1, alpha=0.5,interpolate=True)
ax2.plot(x,y1, c='b')
ax2.set_title('portfolio return($r_2>r_1$)',fontsize=20, y=1.20)

ax3 = axisartist.Subplot(fig, 133)  
#将绘图区对象添加到画布中
fig.add_axes(ax3)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax3.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax3.axis["x"] = ax3.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax3.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax3.axis["y"] = ax3.new_floating_axis(1,0)
ax3.axis["y"].set_axisline_style("-|>", size = 1.0)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.annotate("$w_1'$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax3.annotate('$r$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
ax3.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax3.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax3.annotate('$r_2$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,1,0.1)
#生成sigmiod形式的y数据
y1 = 1.0
y2=x-100
#设置x、y坐标轴的范围
ax3.set_xlim([-5, 5])


ax3.set_ylim(-5, 5)
ax3.hlines(y=1, color='b',xmin=0, xmax=1, linestyle='-')
ax3.fill_between(x,y1, where=y1>y2,color='grey', alpha=0.5,interpolate=True)
ax3.set_title('portfolio return($r_2=r_1$)',fontsize=20, y=1.20)
fig.show()
fig.savefig('Picture4',dpi=150)
#%%
import mpl_toolkits.axisartist as axisartist
#创建画布
fig = plt.figure(figsize=(5, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate("$w_1$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax.annotate('$w_2$', xy=(0.45, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
ax.annotate("$-k/r_1$", xy=(0.4, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate("$k/VaR_\\alpha(r_2)$", xy=(0.35, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
x = np.arange(-5,5,0.1)
#生成sigmiod形式的y数据
y=0.25*x+0.5
y1=x-100
#设置x、y坐标轴的范围
plt.xlim(-5,5)
plt.ylim(-5, 5)

plt.fill_between(x,y,y1, where=y>y1,color='grey', alpha=0.5,interpolate=True)
plt.plot(x,y, c='b')
plt.title('Assets allocation',fontsize=20, y=1.080)
plt.savefig('asset allocation', dpi=150)
#%%asset allocation for w1
fig = plt.figure(figsize=(5, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)  
#将绘图区对象添加到画布中
fig.add_axes(ax)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis[:].set_visible(False)

#ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"] = ax.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1,0)
ax.axis["y"].set_axisline_style("-|>", size = 1.0)
ax.set_xticks([])
ax.set_yticks([])
ax.annotate("$w_1$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=20)
ax.annotate('$w_2$', xy=(0.45, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
ax.annotate("$-k/r_1$", xy=(0.4, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax.annotate("$k/VaR_\\alpha(r_2)$", xy=(0.35, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
x = np.arange(-5,5,0.1)
#生成sigmiod形式的y数据
y=0.25*x+0.5
#设置x、y坐标轴的范围
plt.xlim(-5,5)
plt.ylim(-5, 5)

plt.fill_between(x,y, where=x>0,color='grey', alpha=0.5,interpolate=True)
plt.plot(x,y, c='b')
plt.title('Assets allocation with no short',fontsize=20, y=1.080)
plt.savefig('Asset allocation with no short', dpi=150)
#%%optimal investment
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

fig = plt.figure(figsize=(10, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax1 = axisartist.Subplot(fig, 121)  
#将绘图区对象添加到画布中
fig.add_axes(ax1)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax1.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax1.axis["x"] = ax1.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax1.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax1.axis["y"] = ax1.new_floating_axis(1,0)
ax1.axis["y"].set_axisline_style("-|>", size = 1.0)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.annotate("$VaR_\\alpha(r_2)$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax1.annotate('$w_1^*$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
#ax1.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax1.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
#ax1.annotate('$r_1$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,5,0.1)
#生成sigmiod形式的y数据
y=(10*x-5)/(x+0.2)
#设置x、y坐标轴的范围
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])
ax1.plot(x,y, c='b')
ax1.set_title('Optimal investment in bonds',fontsize=15, y=1.08)

#################################
ax2 = axisartist.Subplot(fig, 122)  
#将绘图区对象添加到画布中
fig.add_axes(ax2)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax2.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax2.axis["x"] = ax2.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax2.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax2.axis["y"] = ax2.new_floating_axis(1,0)
ax2.axis["y"].set_axisline_style("-|>", size = 1.0)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.annotate("$VaR_\\alpha(r_2)$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax2.annotate('$w_2^*$', xy=(0.5, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
#ax1.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax2.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
#ax1.annotate('$r_1$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,5,0.1)
#生成sigmiod形式的y数据
y2=7/(x+0.2)
#设置x、y坐标轴的范围
ax2.set_xlim([-5, 5])
ax2.set_ylim([-5, 5])
ax2.plot(x,y2, c='b')
ax2.set_title('Optimal investment in stocks',fontsize=15, y=1.08)
fig.savefig('picture7', dpi=150)
#%% OPTIMAL INVESTMENT WITH RISK FREE RATE
fig = plt.figure(figsize=(10, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax1 = axisartist.Subplot(fig, 121)  
#将绘图区对象添加到画布中
fig.add_axes(ax1)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax1.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax1.axis["x"] = ax1.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax1.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax1.axis["y"] = ax1.new_floating_axis(1,0)
ax1.axis["y"].set_axisline_style("-|>", size = 1.0)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.annotate("$r_1$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax1.annotate('$w_1^*\'$', xy=(0.48, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
ax1.annotate("$r_2^0$", xy=(0.75, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax1.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax1.annotate('$1$', xy=(0.48, 0.75), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,1,0.1)
#生成sigmiod形式的y数据
y=0.1/(x+0.3)
#设置x、y坐标轴的范围
ax1.set_xlim([-2, 2])
ax1.set_ylim([-2, 2])
ax1.plot(x,y,c='b')
ax1.hlines(y=1, xmin=1, xmax=2, linewidth=2, color='b')
ax1.set_title('Optimal investment in bonds',fontsize=15, y=1.08)

#################################
ax2 = axisartist.Subplot(fig, 122)  
#将绘图区对象添加到画布中
fig.add_axes(ax2)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax2.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax2.axis["x"] = ax2.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax2.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax2.axis["y"] = ax2.new_floating_axis(1,0)
ax2.axis["y"].set_axisline_style("-|>", size = 1.0)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.annotate("$r_1$", xy=(1, 0.5), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax2.annotate('$w_2^*\'$', xy=(0.48, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
ax2.annotate("$r_2^0$", xy=(0.75, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax2.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax2.hlines(y=0, xmin=1, xmax=2, linewidth=2, color='b')
#ax2.annotate('$1$', xy=(0.48, 0.75), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,1,0.1)
#生成sigmiod形式的y数据
y2=(0.2+x)/(x+0.3)
#设置x、y坐标轴的范围
ax2.set_xlim([-2, 2])
ax2.set_ylim([-2, 2])
ax2.plot(x,y2, c='b')
ax2.set_title('Optimal investment in stocks',fontsize=15, y=1.08)
plt.show()
fig.savefig('picture8', dpi=150)
#%%for sigma
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

fig = plt.figure(figsize=(10, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax1 = axisartist.Subplot(fig, 121)  
#将绘图区对象添加到画布中
fig.add_axes(ax1)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax1.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax1.axis["x"] = ax1.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax1.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax1.axis["y"] = ax1.new_floating_axis(1,0)
ax1.axis["y"].set_axisline_style("-|>", size = 1.0)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.annotate("$\\sigma$", xy=(1, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax1.annotate("$w_1^*'$", xy=(0.48, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
#ax1.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax1.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
#ax1.annotate('$r_1$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0.363636,5,0.1)
#生成sigmiod形式的y数据
y=(1.65*x-0.6)/(1.65*x-0.3)
#设置x、y坐标轴的范围
ax1.set_xlim([-5, 5])
ax1.set_ylim([-1, 1])
ax1.plot(x,y, c='b')
ax1.set_title('Optimal investment in bonds',fontsize=15, y=1.08)

#################################
ax2 = axisartist.Subplot(fig, 122)  
#将绘图区对象添加到画布中
fig.add_axes(ax2)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax2.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax2.axis["x"] = ax2.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax2.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax2.axis["y"] = ax2.new_floating_axis(1,0)
ax2.axis["y"].set_axisline_style("-|>", size = 1.0)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.annotate("$\\sigma$", xy=(1, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax2.annotate("$w_2^*'$", xy=(0.48, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
#ax1.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax2.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
#ax1.annotate('$r_1$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0.3636,5,0.1)
#生成sigmiod形式的y数据
y2=0.3/(1.65*x-0.3)
#设置x、y坐标轴的范围
ax2.set_xlim([-5, 5])
ax2.set_ylim([-1, 1])
ax2.plot(x,y2, c='b')
ax2.set_title('Optimal investment in stocks',fontsize=15, y=1.08)
fig.savefig('picture11', dpi=150)
#%%for stocks return
fig = plt.figure(figsize=(10, 5))
#使用axisartist.Subplot方法创建一个绘图区对象ax
ax1 = axisartist.Subplot(fig, 121)  
#将绘图区对象添加到画布中
fig.add_axes(ax1)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax1.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax1.axis["x"] = ax1.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax1.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax1.axis["y"] = ax1.new_floating_axis(1,0)
ax1.axis["y"].set_axisline_style("-|>", size = 1.0)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.annotate("$\\mu$", xy=(1, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax1.annotate("$w_1^*'$", xy=(0.48, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
#ax1.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax1.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
#ax1.annotate('$r_1$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,0.295,0.001)
#生成sigmiod形式的y数据
y=(0.295-x)/(0.595-x)
#设置x、y坐标轴的范围
ax1.set_xlim([-2, 2])
ax1.set_ylim([-1, 1])
ax1.hlines(y=0, xmin=0.295, xmax=2, linewidth=2, color='b')
ax1.plot(x,y, c='b')
ax1.set_title('Optimal investment in bonds',fontsize=15, y=1.08)

#################################
ax2 = axisartist.Subplot(fig, 122)  
#将绘图区对象添加到画布中
fig.add_axes(ax2)
#通过set_visible方法设置绘图区所有坐标轴隐藏
ax2.axis[:].set_visible(False)
#ax.new_floating_axis代表添加新的坐标轴
ax2.axis["x"] = ax2.new_floating_axis(0,0)
#给x坐标轴加上箭头
ax2.axis["x"].set_axisline_style("->", size = 1.0)
#添加y坐标轴，且加上箭头
ax2.axis["y"] = ax2.new_floating_axis(1,0)
ax2.axis["y"].set_axisline_style("-|>", size = 1.0)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.annotate("$\\mu$", xy=(1, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=10)
ax2.annotate("$w_2^*'$", xy=(0.48, 1), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=10)
#ax1.annotate("1", xy=(0.6, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
ax2.annotate("0", xy=(0.5, 0.48), ha='right', va='top', xycoords='axes fraction', fontsize=15)
#ax1.annotate('$r_1$', xy=(0.48, 0.6), xytext=(-15,2), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=20)
x = np.arange(0,0.295,0.001)
#生成sigmiod形式的y数据
y2=0.3/(0.595-x)
#设置x、y坐标轴的范围
ax2.set_xlim([-2, 2])
ax2.set_ylim([-1, 1])
ax2.plot(x,y2, c='b')
ax2.hlines(y=0.9999, xmin=0.295, xmax=2, linewidth=2, color='b')
ax2.set_title('Optimal investment in stocks',fontsize=15, y=1.08)
fig.savefig('picture12', dpi=150)