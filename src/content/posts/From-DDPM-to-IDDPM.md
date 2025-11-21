---
title: From-DDPM-to-IDDPM
published: 2025-09-28
description: From Denoising Diffusion Probabilistic Models (DDPM) to Improved Denoising Diffusion Probabilistic Models (IDDPM)
tags: [Diffusion, GenerativeModels, ComputerVision]
category: ComputerVision
draft: false
---

# Abstraction

本文简要介绍了扩散模型的初期发展，从Denoising Diffusion Probabilistic Models (DDPM)，到Improved Denoising Diffusion Probabilistic Models (IDDPM)。由于扩散模型技术已相对成熟，此处不再给出具体代码实现，力求归纳总结扩散模型初期发展路线。

# DDPM

DDPM由UC Berkeley研究人员在2020年6月发表在arxiv，并被同年的NeurIPS 2020接受。

## Abstraction

扩散模型基本思路是，向图像逐渐加入高斯噪声，使像素分布趋近高斯分布。如果该过程可逆，每次从图像中去除部分高斯噪声，那么就能从高斯噪声生成图像。

更形象地，扩散模型包括两个过程，分别是前向过程 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 和反向过程 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 。我们设过程包括 $T$ 步，生成图像为 $\mathbf{x}_0$ ,高斯噪声为 $\mathbf{x}_T$ 。而扩散模型真正学习的，便是从 $\mathbf{x}_T$ 逐步去噪得到 $\mathbf{x}_0$​ 的变换，即 $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ ，其中 $\theta$ 为模型参数。

## Forward

我们首先考虑前向过程 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ ，即向图像中添加高斯噪声。设每步使用的高斯方差为 $\beta_t$ ，DDPM中，加噪方式即将图像与标准高斯噪声 $\epsilon_{t-1}\sim\mathcal{N}(0,\mathbf{I})$ 加权求和：

$$
\mathbf{x}_{t}=\sqrt{1-\beta_t}\mathbf{x}_{t-1}+\sqrt{\beta_t}\epsilon_{t-1}
$$

其中 $\beta_t$ 即为每步加噪所使用方差。实际上，出于对自然扩散的模拟以及训练效率的优化，通常在起始时使用较小的方差，随步数增加。在DDPM中，作者使用从 $\beta_1=10^{-4}$ 线性增加到 $\beta_T=0.02$ 的高斯方差。

上式中可以证明 $\mathbf{x}_t$ 满足均值为 $\sqrt{1-\beta_t}\mathbf{x}_{t-1}$ ，标准差为 $\sqrt{\beta_t}\mathbf{I}$ 的高斯分布。将上式改写为条件概率分布形式：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})
$$

可见，每个时间步的 $\mathbf{x}_t$ 仅与 $\mathbf{x}_{t-1}$ 有关，满足马尔可夫性质。将公式中 $\mathbf{x}_{t-1}$ 进一步展开：

$$
\mathbf{x}_t=\sqrt{(1-\beta_t)(1-\beta_{t-1})}\mathbf{x}_{t-2}+\sqrt{(1-\beta_t)\beta_{t-1}}\epsilon_{t-2}+\sqrt{\beta_t}\epsilon_{t-1}
$$

注意到 $\epsilon_{t-2}$ 和 $\epsilon_{t-1}$ 同分布，进行合并并设 $\alpha_t=1-\beta_t$ ， $\bar{\alpha}_t=\prod_{i=1}^t\alpha_i$ ，整理得到：

> 由于 $\epsilon\sim\mathcal{N}(0,\mathbf{I})$ ，后文中若无特别需要，不再以 $\epsilon_t$ 等形式加以区分。

$$
\mathbf{x}_t=\sqrt{\bar{\alpha}_t}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_t}\epsilon
$$

通过上述推导，给定 $\mathbf{x}_0$ 与时间步，可以直接得到 $\mathbf{x}_t$ ，而无需经过 $t$ 步加噪。同理将上式改写为：

$$
q(\mathbf{x}_t|\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I})
$$

可见， $\mathbf{x}_t$ 可视作原始图像 $\mathbf{x}_0$ 与高斯噪声 $\epsilon$ 的线性组合，且组合系数平方和为1。

## Backward

反向过程即从 $\mathbf{x}_T$ 逐步去噪得到 $\mathbf{x}_0$ ，也即学习 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 。根据贝叶斯公式：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t)=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1})q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}
$$

上式中， $q(\mathbf{x}_{t-1})$ 与 $q(\mathbf{x}_t)$ 仍然未知。在贝叶斯公式中附加 $\mathbf{x}_0$ 作为条件，得到：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
$$

由于先验分布 $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ 是马尔可夫过程，DDPM在此处假设 $\mathbf{x}_t$ 仅与 $\mathbf{x}_{t-1}$ 相关，即：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)=q(\mathbf{x}_t|\mathbf{x}_{t-1})
$$

到此为止，上式中仅 $\mathbf{x}_0$ 仍然未知，但易证上式中条件概率分布均满足高斯分布：

$$
\begin{aligned}
q(\mathbf{x}_t|\mathbf{x}_{t-1})&=\mathcal{N}(\mathbf{x}_t;\sqrt{\alpha_t}\mathbf{x}_{t-1},1-\alpha_t)\\
q(\mathbf{x}_{t-1}|\mathbf{x}_0)&=\mathcal{N}(\mathbf{x}_{t-1};\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0,1-\bar{\alpha}_{t-1})\\
q(\mathbf{x}_t|\mathbf{x}_0)&=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,1-\bar\alpha_t)
\end{aligned}
$$

将上式展开并忽略常数项，整理为关于 $\mathbf{x}_{t-1}$ 的多项式，得到：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\propto\exp\left(-\frac{1}{2}\left[\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)\mathbf{x}_{t-1}^2-\left(\frac{2\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t+\frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0\right)\mathbf{x}_{t-1}+C(\mathbf{x}_t,\mathbf{x}_0)\right]\right)
$$

可以证明反向过程的分布同样满足高斯分布，对比指数部分展开得到：

$$
\begin{aligned}
\frac{1}{\sigma_t^2}&=\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\\
\frac{2\mu_t}{\sigma_t^2}&=\frac{2\sqrt{\alpha_t}}{\beta_t}\mathbf{x}_t+\frac{2\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}\mathbf{x}_0
\end{aligned}
$$

求解得到均值 $\mu_t$ 与标准差 $\sigma_t$ ：

$$
\begin{aligned}
\sigma_t&=\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)^{-1/2}\\
\mu_t&=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0
\end{aligned}
$$

进一步的，将前向过程中 $\mathbf{x}_0$ 化简，其中前向过程中加入的高斯噪声 $\epsilon$ 为未知项，将使用去噪网络估计，设 $\tilde{\epsilon}=\epsilon_{\theta}(\mathbf{x}_t,t)$ 为去噪网络输出，得到：

$$
\mu_t=\frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\tilde{\epsilon}\right)
$$

> 为了书写便利以及不引起歧义，后文中若无特别需要，均以 $\tilde{\epsilon}$ 表示去噪网络输出，即 $\epsilon_{\theta}(\mathbf{x}_t,t)$ 。

到此为止，反向过程分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ 仅存在 $\tilde{\epsilon}$ 一项未知，也即去噪网络学习的目标。

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_{t-1};\mu_t,\sigma_t^2\mathbf{I})
$$

注意，在去噪网络中，一般还需输入时间步 $t$ ，因为前向过程中 $\mathbf{x}_t$ 的噪声含量由 $t$ 决定的，在预测噪声时也需明确时间步 $t$​ 作为参考。

> 特别的，如果将 $\sigma_t^2$ 整理，可以得到有关 $\beta_t$ 的公式如下，称为 $\tilde{\beta}_t$ ：

$$
\sigma_t^2=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

> 在DDPM中作者设定 $\sigma_t^2=\beta_t$ 与 $\sigma_t^2=\tilde{\beta_t}$ ，通过对比实验得到差别较小。而后续工作如IDDPM中，将 $\beta_t$ 设为可学习参数。

## Training

DDPM论文中给出训练过程如下：
1. 从数据集 $q(\mathbf{x}_0)$ 中采样得到 $\mathbf{x}_0$
2. 从 $1$ 到 $T$ 的均匀分布中采样得到 $t$
3. 从标准高斯分布中采样得到 $\epsilon$
4. 根据 $\mathbf{x}_0$ 与 $\epsilon$ 加权求和得到 $\mathbf{x}_t$ ，随后将 $\mathbf{x}_t$ 与 $t$ 输入到去噪网络预测 $\tilde{\epsilon}$ ，利用 $\epsilon$ 计算 $L_2$ 损失进行优化

$$
\nabla_{\theta}{||\epsilon -\tilde{\epsilon}||}^2
$$

5. 重复直至模型收敛

## Sample

DDPM论文中，给出采样步骤如下：
1. 从标准正态分布中采样得到 $\mathbf{x}_T$
2. 重复 $T$ 步去噪过程，得到 $\mathbf{x}_0$

## Summary

到此为止我们回顾了DDPM的理论，但显然相比于GAN、VAE等早期方法，DDPM的采样过程是相当低效的；同时在FID、Inception Score等量化指标上，DDPM也未能达到sota。

尽管如此，DDPM还是唤起了社区对Diffusion的研究热情，在2020年发表在NeurIPS后，相当多的Diffusion工作开始涌现。DDIM便是其中一员。

# DDIM

DDIM由Stanford研究人员在2020年10月发表在arxiv，并被ICLR 2021接受。

## Abstraction

DDPM反向过程中，目标即计算概率分布 $q(\mathbf{x}_{t-1}|\mathbf{x}_t)$ 。DDPM 利用贝叶斯公式将其变形为：

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)=\frac{q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
$$

实际上，目前未知的分布为 $q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)$ 。DDPM作马尔科夫假设，认为 $\mathbf{x}_t$ 仅与 $\mathbf{x}_{t-1}$ 有关，即：

$$
q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)=q(\mathbf{x}_t|\mathbf{x}_{t-1})
$$

可见DDPM的缺点在于去噪仅能在相邻时间步之间进行。如果能够导出 $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$ ，其中 $\tau$ 为某中间时间步，即可实现采样加速。

于是，DDIM的出发点即为：

1. 保持前向过程分布 $q(\mathbf{x}_t|\mathbf{x}_{t-1})=\mathcal{N}\left(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I}\right)$ 不变
2. 推导不依赖于马尔可夫假设分布 $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$

## Derivation of $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$

> 为了防止混淆，此处使用通用的符号 $\tau\in(0,t)$ 表示可能的中间时间步，同时其 $\mu_t,\sigma_t$ 均表示 $q(\mathbf{x}_{\tau}|\mathbf{x}_t,\mathbf{x}_0)$ 所对应均值与方差。特别的，在DDIM原文中使用了与DDPM不一致的符号体系。为了保证一致性，这里依然使用DDPM符号约定，即：

$$
\begin{aligned}
\alpha_t&=1-\beta_t\\
\bar{\alpha}_t&=\prod_{i=1}^t\alpha_i
\end{aligned}
$$

回顾DDPM中导出的结论， $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$ 为高斯分布，其均值与方差为：

$$
\begin{aligned}
\sigma_t&=\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)^{-1/2}\\
\mu_t&=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t+\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0
\end{aligned}
$$

可见均值为 $\mathbf{x}_0$ 与 $\mathbf{x}_t$ 的线性组合，而方差为时间步的函数。DDIM假设 $q(\mathbf{x}_{\tau}|\mathbf{x}_t,\mathbf{x}_0)$ 仍为高斯分布并满足上述规律，使用待定系数法：

$$
q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_\tau;\lambda_t\mathbf{x}_0+k_t\mathbf{x}_t,\sigma_t^2\mathbf{I})
$$

也即 $\mathbf{x}_\tau=\lambda_t\mathbf{x}_0+k_t\mathbf{x}_t+\sigma_t\epsilon$ 。将前向过程代回，得到：

$$
\mathbf{x}_\tau=(\lambda_t+k_t\sqrt{\bar{\alpha}_t})\mathbf{x}_0+\sqrt{k_t^2(1-\bar{\alpha}_t)+\sigma_t^2}\epsilon
$$

而根据前向过程，有 $\mathbf{x}_\tau=\sqrt{\bar{\alpha}_\tau}\mathbf{x}_0+\sqrt{1-\bar{\alpha}_\tau}\epsilon$ ，于是得到方程组如下：

$$
\begin{aligned}
\lambda_t+k_t\sqrt{\bar{\alpha}_t}&=\sqrt{\bar{\alpha}_\tau}\\
\sqrt{k_t^2(1-\bar{\alpha}_t)+\sigma_t^2}&=\sqrt{1-\bar{\alpha}_\tau}
\end{aligned}
$$

解方程组得到 $\lambda_t$ 与 $k_t$ 如下：

$$
\begin{aligned}
\lambda_t&=\sqrt{\bar{\alpha}_\tau}-\sqrt{\frac{(1-\bar{\alpha}_\tau-\sigma_t^2)\bar{\alpha}_t}{1-\bar{\alpha}_t}}\\
k_t&=\sqrt{\frac{1-\bar{\alpha}_\tau-\sigma_t^2}{1-\bar{\alpha}_t}}
\end{aligned}
$$

到此为止，已知 $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$ 均值参数，而方差 $\sigma_t^2$ 未知。

## Value of $\sigma_t^2$

根据推导，推导结果对应于一组解：通过规定不同的方差，可以得到不同的采样结果。

最后，将 $\mathbf{x}_0$ 以 $\mathbf{x}_t$ 替换，同样的以 $\tilde{\epsilon}$ 表示去噪模型输出，得到关于 $\mathbf{x}_t$ 的 $\mu_t$ 表达式：

$$
\mu_t=\sqrt{\bar{\alpha}_\tau}\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\tilde{\epsilon}}{\sqrt{\bar{\alpha}_t}}+\sqrt{1-\bar{\alpha}_\tau-\sigma_t^2}\tilde{\epsilon}
$$

化简得到 $\mathbf{x}_\tau$ 的表达式，与DDIM原文对齐，即：

$$
\mathbf{x}_\tau=\sqrt{\bar{\alpha}_\tau}\underbrace{\frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}_t}\tilde{\epsilon}}{\sqrt{\bar{\alpha}_t}}}_{\mathrm{predicted\ }\mathbf{x}_0}+\underbrace{\sqrt{1-\bar{\alpha}_\tau-\sigma_t^2}\tilde{\epsilon}}_{\mathrm{direction\ pointing\ to\ }\mathbf{x}_t}+\underbrace{\sigma_t\epsilon}_{\mathrm{random\ noise}}
$$

实际上，由于 $\sigma_t$ 的不确定性，实际上上式为 $\mathbf{x}_\tau$ 的一组解。一个很自然的想法便是，参考DDPM中对 $\sigma_t$ 的推导。作者给出了一个 $\sigma_t$ 的形式如下：

$$
\sigma_t=\eta\sqrt{\frac{1-\bar{\alpha}_\tau}{1-\bar{\alpha}_t}}\sqrt{1-\alpha_t}
$$

1. 当 $\eta=1$ ，即整体形式趋向DDPM
2. 当 $\eta=0$ ，此时生成过程将不再添加随机噪声，采样的过程变为确定：每个 $\mathbf{x}_T$ 对应唯一 $\mathbf{x}_0$

## Acceleration of Sampling

到此为止，已经形式化的推导得到了 $q(\mathbf{x}_\tau|\mathbf{x}_t,\mathbf{x}_0)$ ，DDIM将在此基础上，将时间步从 $T$ 减少至 $S$ ，抽取子序列 $[\tau_S,\tau_{S-1},...,\tau_2,\tau_1]$ 进行采样。

在DDIM的附录中，给出了两种子序列的选取方式：

1. 线性选取：令 $\tau_i=\lfloor c_i\rfloor$
2. 二次方选取：令 $\tau_i=\lfloor c_i^2\rfloor$

其中 $c$ 为常量，只需保证 $\tau_{-1}$ 也即最后采样时间步尽可能与 $T$ 接近。

> 在原文的实验中，CIFAR10使用二次方选取，其他数据集均采用线性选取方式。

## Features of DDIM

到此为止我们讲完了DDIM的故事，总结其相比于DDPM具有的特性如下：

1. 采样一致性：DDIM的采样是确定的，其结果只受 $\mathbf{x}_T$ 影响。经过实验验证，对于同一 $\mathbf{x}_T$ 使用不同采样过程，将生成相近的 $\mathbf{x}_0$ 。因此 $\mathbf{x}_T$ 在一定程度上可以视作 $\mathbf{x}_0$ 的嵌入。

    > 一致性可为生成图像提供了trick。在初始选取较小的时间步数量生成较为粗糙的图像，若符合预期，再使用较大时间步数量进行精细生成。

2. 语义插值效应： $\mathbf{x}_T$ 可以视作 $\mathbf{x}_0$ 的嵌入，则应当同样具有隐概率模型的语义差值效应。实验首先选取隐变量 $\mathbf{x}_T^{(0)}$ 与 $\mathbf{x}_T^{(1)}$ ，分别采样得到结果。随后使用球面线性插值得到一系列中间隐变量。其中插值定义为：

    $$
    \mathbf{x}_T^{(\alpha)}=\frac{\sin((1-\alpha)\theta)}{\sin(\theta)}\mathbf{x}_T^{(0)}+\frac{\sin(\alpha\theta)}{\sin(\theta)}\mathbf{x}_T^{(1)}
    $$

    其中 $\theta=\arccos\left(\frac{(\mathbf{x}_T^{(0)})^T\mathbf{x}_T^{(1)}}{||\mathbf{x}_T^{(0)}||~||\mathbf{x}_T^{(1)}||}\right)$ 。

## Summary

DDIM的推导是比较优美的，结果也是如此。最重要的是对采样速度有所提升。DDIM和IDDPM均包含了对DDPM采样速度进行优化的尝试，相比之下DDIM解决的方式更加彻底，这导致实际上部分模型在采样时遵循了DDIM的路线，如Stable Diffusion。

而区别于DDIM，IDDPM主要针对DDPM的训练过程进行改进，主要包括两个方面：

1. 替换DDPM中固定方差为可学习的方差
2. 加噪过程使用余弦形式Scheduler

# IDDPM

IDDPM由OpenAI研究人员在2021年2月发表在arxiv，并被同年的ICML 2021接受。

## Abstraction

回顾DDPM中，设定一系列固定 $\beta_t$ 作为加噪方差，并将 $\sigma^2_t$ 分别取为：

$$
\begin{aligned}
\sigma_t^2&=\beta_t\\
\sigma_t^2&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
\end{aligned}
$$

IDDPM复现了这一实验并发现，仅在扩散过程最开始时两者有较大的差距，而当步骤增大时，两者基本上没有区别：当步骤足够大时，$\sigma_t$ 的选取对采样的质量影响不大。

> 换言之， $\mu_t$ 比方差 $\sigma_t^2$ 更能决定生成的分布。

作者另一发现是，在扩散过程中最初的几步扩散对VLB影响最大，而在这里 $\sigma_t$ 依然有一定的作用。由此，作者认为可以通过选取方差来获得更好的对数似然，由此引入可学习方差。

## Learnable Variance

IDDPM作者认为方差变化范围较小，不易用神经网络进行学习，故实际上使用方差是对 $\beta_t$ 与 $\tilde{\beta}_t$ 进行插值的结果：

$$
\sigma_t=\exp(v\log\beta_t+(1-v)\log\tilde{\beta}_t)
$$

实际训练时并未对 $v$ 进行约束，理论上来说方差未必在这两者之间，不过最终实验未发现上述情况。

## Cosine Schedule

作者还发现线性 $\beta_t$ 对于低分辨率的图像表现不佳。如果最开始时候加入较大噪声，会严重破坏图像信息，从而不利于学习。

考虑到低分辨率图像包含的信息本身较少，线性schedule对于低分辨率图像来说仍然加噪较快。作者将方差通过cosine形式定义，不过并非直接定义 $\beta_t$ ，而是 $\bar{\alpha}_t$ ：

$$
\begin{aligned}
\bar{\alpha}_t&=\frac{f(t)}{f(0)}\\
f(t)&=\cos\left(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\right)^2
\end{aligned}
$$

可见schedule在 $t=0$ 与 $t=T$ 附近都变化较小，相比linear schedule对信息破坏更慢。这也印证了在扩散开始的时候缓慢加噪将得到更好的训练效果。

除此之外，作者还有部分比较细节的考虑，如选取较小的偏移量 $s=8\times10^{-3}$ 防止 $\beta_t$ 在 $t=0$ 附近过小，并将 $\beta_t$ 裁剪到 $0.999$ 防止 $t=T$ 附近出现奇异点。具体在Implementation将体现。

## Training

IDDPM训练使用的损失为两项损失加权：

$$
L_\mathrm{hybrid}=L_\mathrm{simple}+\lambda L_\mathrm{vlb}
$$

其中， $L_\mathrm{simple}$ 即DDPM中的L2损失： 

$$
L_\mathrm{simple}=E_{t,\mathbf{x}_0,\epsilon}(||\epsilon-\tilde{\epsilon}||^2)
$$

而 $L_\mathrm{vlb}$ 为VAE形式损失函数：

$$
\begin{aligned}
L_\mathrm{vlb}&=L_0+L_1+\cdots+L_{T-1}+L_T\\
L_0&=-\log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\\
L_{t-1}&=D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)||p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))\\
L_T&=D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0)||p(\mathbf{x}_T))
\end{aligned}
$$

其中为了防止 $L_\mathrm{vlb}$ 影响 $L_\mathrm{simple}$ ，这里使用了较小的权重 $\lambda=1\times10^{-3}$ 并对VLB损失中均值项 $\mu_t$ 进行 stop-gradient，从而让 $L_\mathrm{simple}$ 依然是均值的主要决定因素。

原文中，作者本想直接优化 $L_\mathrm{vlb}$ ，但是分析认为 $L_\mathrm{vlb}$ 的梯度比 $L_\mathrm{hybrid}$ 更加noisy。

这是因为不同时间步VLB损失大小不一，均匀采样时间步 $t$ 会引入比较多的噪音。为了解决这一问题，作者引入重要性采样：

$$
L_\mathrm{vlb}=E_{t\sim p_t}\left[\frac{L_t}{p_t}\right],\text{where}~p\propto\sqrt{E[L_t^2]}~\text{and}~\sum p_t=1
$$

其中由于 $E[L_t^2]$ 未知且在训练过程中变化，实际上训练时候会保存10项历史损失，并对此进行动态更新。

除此之外，这些损失还能用于计算每个时间步的权重，在计算最终损失时每个时间步损失先乘以对应的权重，再进行加和得到整体损失。

## Summary

IDDPM相比DDIM，主要进行工程上的优化。这些工作奠定了Diffusion在生成领域的地位。

# References

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
2. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
3. [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)