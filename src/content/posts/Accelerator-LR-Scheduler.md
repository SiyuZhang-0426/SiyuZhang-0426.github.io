---
title: Accelerator-LR-Scheduler
published: 2026-03-06
description: Accelerate 分布式训练中 LR Scheduler 的步数设定与常见误区
tags: [Accelerator, DeepSpeed, Tools]
category: Tools
draft: false
---

# 摘要

这篇文章讨论一个常见但容易混淆的问题：在使用 Accelerate 进行多卡训练时，LR Scheduler 的 `step` 计数设定到底是什么，以及为什么错误配置会导致学习率曲线异常。

核心结论是：如果你采用 Accelerate 的默认机制，Scheduler 初始化应遵循单卡设定；否则在多卡下会出现总步数被重复放大、学习率提前或延后衰减等问题。

# 现象与问题定位

在阅读 CogVideoX 源码时，我在 `trainer.py` 的 [`prepare_optimizer`](https://github.com/zai-org/CogVideo/blob/main/finetune/trainer.py#L296) 中看到如下逻辑：

```py
num_update_steps_per_epoch = math.ceil(len(self.data_loader) / self.args.gradient_accumulation_steps)
if self.args.train_steps is None:
    self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
    self.state.overwrote_max_train_steps = True

use_deepspeed_lr_scheduler = (
    self.accelerator.state.deepspeed_plugin is not None
    and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
)
total_training_steps = self.args.train_steps * self.accelerator.num_processes
num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

if use_deepspeed_lr_scheduler:
    from accelerate.utils import DummyScheduler

    lr_scheduler = DummyScheduler(
        name=self.args.lr_scheduler,
        optimizer=optimizer,
        total_num_steps=total_training_steps,
        num_warmup_steps=num_warmup_steps,
    )
else:
    lr_scheduler = get_scheduler(
        name=self.args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
        num_cycles=self.args.lr_num_cycles,
        power=self.args.lr_power,
    )
```

随后在 `prepare_for_training` 中，各模块才会被 `accelerator` 包装。也就是说，上面代码执行时，`data_loader` 还没有经过 Accelerate 的分布式封装，此时 `len(self.data_loader)` 仍是单卡设定：

$$
len\_{data\_loader} = \frac{dataset\_size}{micro\_batch\_size}
$$

但后续 `total_training_steps` 和 `num_warmup_steps` 又显式乘了 `self.accelerator.num_processes`，导致两者被额外放大 `world_size` 倍。

如果以常见分布式训练直觉来理解，通常会有如下的关系：

$$
len\_{data\_loader} = \frac{dataset\_size}{micro\_batch\_size * world\_size}
\\
total\_training\_steps = train\_epochs * steps\_per\_epoch * grad\_acc\_steps
$$

然而在 Accelerate 的默认设计下，Scheduler 的“步数推进”并不完全等价于我们直觉中的“每次 optimizer.step 推进一步”。如果同时采用单卡口径的 dataloader 长度，又再手动乘 `world_size`，就会形成重复修正，学习率轨迹偏离预期。官方 demo 也可观察到这一现象。

# 外部讨论与结论一致性

我先查阅了 CogVideoX 的 issue，发现 [issue#747](https://github.com/zai-org/CogVideo/issues/747) 中已有类似反馈。

以下是作者的回复：

```plaintext
Thank you for pointing it out. Indeed, accelerate may handle settings for multi-card parallelism on its own, so the scheduler should be configured according to single-card settings before preparing the scheduler.
...
Thank you again for pointing it out, and you are welcome to submit a PR to fix this error.
```

这说明如果训练代码遵循 `single-card settings` 初始化 Scheduler，那么 `total_training_steps` 和 `num_warmup_steps` 不应再手动乘 `num_processes`。否则会导致调度终点错位，例如训练结束时 LR 停在 `learning_rate / world_size`，而不是预期的 `0`。

在 Hugging Face discuss 中也有同类讨论：[Learning Rate Scheduler Distributed Training](https://discuss.huggingface.co/t/learning-rate-scheduler-distributed-training/30453)。

开发者通过 warmup 过程观察到：多卡下一次 `lr_scheduler.step()` 可能对应内部推进 `world_size` 次，而单卡只推进一次。

Accelerate 官方文档 [Comparing performance across distributed setups](https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates) 也给出了对应说明：

```plaintext
You will also find that accelerate will step the learning rate based on the number of processes being trained on. This is because of the observed batch size noted earlier. So in the case of 2 GPUs, the learning rate will be stepped twice as often as a single GPU to account for the batch size being twice as large (if no changes to the batch size on the single GPU instance are made).
```

这里的 `step` 容易被误读。更准确地说，在 Accelerate 的语境里，它强调的是 Scheduler 的推进语义，而不是简单等同于外部可见的 `optimizer.step()` 调用次数。

这与 Accelerate 在 [index](https://huggingface.co/docs/accelerate/index) 中强调的设计目标一致：

```
Accelerate is a library that enables the same PyTorch code to be run across any distributed configuration by adding just four lines of code! In short, training and inference at scale made simple, efficient and adaptable.
```

也就是说，Accelerate 通过“单卡设定初始化 + 框架内部处理分布式差异”的方式，尽量保证同一套训练代码跨分布式配置可复用。

# 结合源码看机制

进一步看 Accelerate 源码。`accelerator.py` 的 [`__init__`](https://github.com/huggingface/accelerate/blob/589fddd317f008e704073c133bc2cb8958f287e6/src/accelerate/accelerator.py#L463) 提供参数 `step_scheduler_with_optimizer`，默认值是 `True`。

该参数会影响封装后的 `AcceleratedScheduler.step_with_optimizer`，其行为体现在 [`scheduler.py#step()`](https://github.com/huggingface/accelerate/blob/589fddd317f008e704073c133bc2cb8958f287e6/src/accelerate/scheduler.py#L54)：

```py
def step(self, *args, **kwargs):
    if not self.step_with_optimizer:
        # No link between scheduler and optimizer -> just step
        self.scheduler.step(*args, **kwargs)
        return

    # Otherwise, first make sure the optimizer was stepped.
    if not self.gradient_state.sync_gradients:
        if self.gradient_state.adjust_scheduler:
            self.scheduler._step_count += 1
        return

    for opt in self.optimizers:
        if opt.step_was_skipped:
            return
    if self.split_batches:
        # Split batches -> the training dataloader batch size is not changed so one step per training step
        self.scheduler.step(*args, **kwargs)
    else:
        # Otherwise the training dataloader batch size was multiplied by `num_processes`, so we need to do
        # num_processes steps per training step
        num_processes = AcceleratorState().num_processes
        for _ in range(num_processes):
            # Special case when using OneCycle and `drop_last` was not used
            if hasattr(self.scheduler, "total_steps"):
                if self.scheduler._step_count <= self.scheduler.total_steps:
                    self.scheduler.step(*args, **kwargs)
            else:
                self.scheduler.step(*args, **kwargs)
```

由这段实现可以得到一个关键结论：

1. 当 `step_with_optimizer = True` 且 `split_batches = False` 时，封装后的每次 `lr_scheduler.step()` 在内部会推进约 `world_size` 次。
2. 当 `step_with_optimizer = False` 时，每次 `lr_scheduler.step()` 只推进一次，不会做上述按进程数展开。

需要强调：这里的“单卡设定/多卡设定”说的是初始化与调度设定，不是是否真的只使用一张卡训练。

# 实践建议

基于以上分析，实践中可以按下面原则配置：

1. 使用 Accelerate 默认调度行为时，Scheduler 相关总步数按单卡设定计算，不手动乘 `world_size`。
2. 若你明确关闭 `step_with_optimizer` 并改为手动控制 Scheduler，则要自行保证步数设定与训练循环一致。
3. 检查 warmup 曲线是否与预期对齐，是定位该类问题最直接的方法。

对于更多讨论，可参考 Accelerate 官方 issue [#662](https://github.com/huggingface/accelerate/issues/662)。

# References

1. [https://github.com/zai-org/CogVideo/blob/main/finetune/trainer.py#L296](https://github.com/zai-org/CogVideo/blob/main/finetune/trainer.py#L296)
2. [https://github.com/zai-org/CogVideo/issues/747](https://github.com/zai-org/CogVideo/issues/747)
3. [https://discuss.huggingface.co/t/learning-rate-scheduler-distributed-training/30453](https://discuss.huggingface.co/t/learning-rate-scheduler-distributed-training/30453)
4. [https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates](https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates)
5. [https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index)
6. [https://github.com/huggingface/accelerate/blob/589fddd317f008e704073c133bc2cb8958f287e6/src/accelerate/accelerator.py#L463](https://github.com/huggingface/accelerate/blob/589fddd317f008e704073c133bc2cb8958f287e6/src/accelerate/accelerator.py#L463)
7. [https://github.com/huggingface/accelerate/blob/589fddd317f008e704073c133bc2cb8958f287e6/src/accelerate/scheduler.py#L54](https://github.com/huggingface/accelerate/blob/589fddd317f008e704073c133bc2cb8958f287e6/src/accelerate/scheduler.py#L54)
8. [https://github.com/huggingface/accelerate/issues/662](https://github.com/huggingface/accelerate/issues/662)
