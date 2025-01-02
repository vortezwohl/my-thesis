<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css" integrity="sha384-Um5gpz1qZLKUhoJ45w8AgWmC+N7MKZARfA4X5Qxv7YOuW+g79o8IKS3+MxQznzB" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex/dist/contrib/auto-render.min.js" integrity="sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5k8vOGmdNQED/202KzynRBf+Cvo9bIiH1EM" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>

# CEO: 基于 ReAct 范式的轻量级自主智能体框架

## 引言

- 研究背景与动机

- 研究目标

## 背景知识

- 大语言模型

    - Transformers 模型与深度神经网络技术

- CoT，ToT 范式

- ReAct 范式

- 工具学习

    - 工具增强学习

    - 工具导向学习

- 自主智能体

    - 智能体四要素

        1. 规划

        2. 记忆

        3. 动作

        4. 工具

    - 工作流与自主智能体的区别

    - 自主智能体实现任务自动化

- 多智能体系统

    - 多智能体通信问题

    - 多智能体任务分配问题

    - 多智能体任务同步问题

## 相关工作

- Openai Swarm

- AutoGen

- CrewAI

## 我的算法

- 工具理解

    基于函数名、函数参数、返回类型、函数内注释、以及函数实现完整源码对函数进行全方位地理解，生成函数 Docstring。

    将函数名、Docstring、参数（包括类型）、返回类型组合为结构化 Json 形式，作为函数的表示形式，并称之为”能力“。

- 自然语言问题理解

    智能体依据自己已拥有的能力，对任务进行按步拆分，并估计完成任务所需步数 expected_steps

- 动作规划

    不同于任务规划，动作规划仅规划智能体的下一动作(Next Move)，任务规划的作用在于基于智能体已有能力预估任务所需总步骤

    1. 记忆与任务分析: 智能体会分析用户提供的任务和智能体记忆中的历史事件。它将列出与任务相关的事件，并逐步提取出与任务直接相关的重要信息。这一过程会帮助智能体了解在执行任务的背景下已完成的步骤和当前的状态。

    2. 任务完成状态评估: 在分析完记忆中的历史事件后，智能体需要判断任务是否已经完全实现。它会比较历史事件中的信息与任务要求，确认哪些步骤已经完成，哪些步骤仍未完成。这一步骤会是判断后续行动的依据。

    3. 能力匹配与未完成任务识别: 一旦确认任务未完全达成，智能体将检查自身可用的能力（Abilities），并分析哪些能力可以用来完成未完成的任务部分。此过程会基于之前对历史事件的分析来决定下一步的具体行动。

    4. 规划下一步行动: 倘若智能体找到了合适的能力以完成未完成的任务，它将制定出一个具体的行动计划，包括选择恰当的能力及其所需的参数。这一过程也包括对选择参数的理由进行解释，以确保行动计划合理可行。

    5. 执行动作: 智能体根据先前规划的下一步动作，进行动作执行，并将执行结果和执行操作进行记忆，以供后续决策参考 (即 ReAct 范式)

- 适时放弃

    如上文所述，在任务规划中，智能体预估了该任务所需步骤需要 expected_steps 步，那如果 expected_steps 步内任务依然没有完成呢？智能体将会无休止地执行下去，你可以通过限制智能体的最大执行次数来解决该问题，但是假如一个问题预估需要10步解决，而你设置的最大执行步骤是5步，那么该问题就永远无法解决，很明显这种方法不够灵活，于是我提出一种基于概率惩罚的适时放弃算法，算法描述如下：

    在动作开始时，智能体会通过基于自身拥有能力(工具)和拆解后的任务目标估计完成任务所需的步数。如果实际步数超过了预期步数，智能体将以概率 p 放弃任务，以概率 1-p 坚持任务。如果智能体选择坚持，那么在下一次执行时，概率 p 将乘以惩罚因子 beta。如果 beta 和 初始 p 较大，它鼓励放弃任务；如果 beta 和 初始 p 较小，它鼓励持续尝试。
    
    概率惩罚递推式：$$p_{\text{new}} = (\beta \cdot p) \mod 1 $$

- 动作执行

    根据智能体对 Next Move 的预测，选择特定工具(能力)，以及 Next Move 预测中给出的参数，进行函数执行并获取反馈。

    反馈包括：
    1. 本次执行了什么动作
    2. 本次使用了什么参数
    3. 动作得到了什么反馈

    反馈示例：I used the ability of a calculator to evaluate a mathematical expression. The choice I made was to calculate the volume of a sphere with a radius of 9.5 using the formula \\((4/3) * \\pi * r^3\\). The expression I input was '(4/3) * 3.14159 * (9.5)**3'. \n\nAfter processing this expression, the result I obtained is 3591.36096833333.
    反馈会被写入智能体记忆中。

- 短期记忆

    记忆以如下结构存储

    ```python
    now = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S.%f')
    new_memory = {
        "date_time": now,
        "agent_name": self._name,
        f"message_from_{self._name}": action_performed
    }
    self._memory[f"{self._name} at {now}"] = new_memory
    ```
    记忆可以在不同智能体之间进行传递，智能体 B 可以通过记忆接力获得智能体 A 的所有记忆

## 实验

...

## 算法优势

- 与其他框架比较

    - 性能：速度、准确性

    - 扩展性，灵活性

    - 易用性，通用性

    - 特定场景下的优势：需要自主决策的场景下

- 创新点

    - ReAct 范式，灵活决策

    - 适时放弃策略

    - 多智能体协作稳定

## 算法应用场景

- 任务自动化

- 多智能体系统

- 智能体研究

## 算法的局限性

## 结论与未来工作
