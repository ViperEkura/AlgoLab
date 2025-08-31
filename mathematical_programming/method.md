设 $y_1, y_2$ 为二进制变量， $z$ 为辅助连续变量，需要建模：

$$
z = x \cdot y_1 \cdot y_2
$$

### 第一步：引入中间二进制变量 $w = y_1 \land y_2$（即 $w = y_1 \cdot y_2$）：
$$
\begin{aligned}
& w \leq y_1 \\
& w \leq y_2 \\
& w \geq y_1 + y_2 - 1 \\
& w \in \{0, 1\}
\end{aligned}
$$

### 第二步：建模 $z = x \cdot w$（其中 $M$ 是一个足够大的常数）：

$$
\begin{aligned}
& z \leq x + M(1 - w) \\
& z \geq x - M(1 - w) \\
& z \leq M w \\
& z \geq 0
\end{aligned}
$$

或者等价地（常见形式）：

$$
\begin{aligned}
& z \leq x \\
& z \leq M w \\
& z \geq x - M(1 - w) \\
& z \geq 0
\end{aligned}
$$


**说明**：  
- 第一步通过线性约束刻画了 $w = y_1 \cdot y_2$（逻辑与）。  
- 第二步通过大M法将乘积 $z = x \cdot w$ 线性化，确保当 $w = 1$ 时 $z = x$，当 $w = 0$ 时 $z = 0$。  
- $M$ 应取 $x$ 的上界（若 $x$ 无界，需根据问题背景选取合适值）。  

这种表示既清晰又符合优化建模的规范。