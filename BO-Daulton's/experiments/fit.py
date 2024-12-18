import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 假设我们有5个点的数据
x = np.array([10, 30, 50, 70, 90])
y = np.array([0.5048, 0.4717, 0.1886, 0.3591, 0.6883])  # 示例数据

# 方法1：使用 numpy 的 polyfit
def np_fit():
    # 进行三阶多项式拟合 (ax^3 + bx^2 + cx + d)
    coefficients = np.polyfit(x, y, 3)
    
    # 生成拟合曲线的点
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = np.polyval(coefficients, x_fit)
    
    return x_fit, y_fit, coefficients

# 方法2：使用 scipy 的 curve_fit
def scipy_fit():
    # 定义三阶多项式函数
    def cubic(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d
    
    # 进行拟合
    popt, _ = curve_fit(cubic, x, y)
    
    # 生成拟合曲线的点
    x_fit = np.linspace(min(x - 5), max(x + 5), 100)
    y_fit = cubic(x_fit, *popt)
    
    return x_fit, y_fit, popt

# 绘制结果
def plot_results():
    plt.figure(figsize=(10, 5))
    
    # 原始数据点
    plt.scatter(x, y, color='red', label='Original Data')
    
    # # 方法1的拟合结果
    # x_fit1, y_fit1, coef1 = np_fit()
    # plt.plot(x_fit1, y_fit1, 'b-', label='NumPy Fit')
    # print(f"NumPy coefficients (ax^3 + bx^2 + cx + d):\n{coef1}")
    
    # 方法2的拟合结果
    x_fit2, y_fit2, coef2 = scipy_fit()
    plt.plot(x_fit2, y_fit2, 'g--', label='SciPy Fit')
    print(f"\nSciPy coefficients (ax^3 + bx^2 + cx + d):\n{coef2}")
    
    fit_data = np.c_[x_fit2, y_fit2]
    np.savetxt('fit_data.txt', fit_data, delimiter = '\t', fmt = '%.4f')    

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results()