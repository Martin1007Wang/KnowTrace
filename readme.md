# KnowTrace

## 项目简介

本仓库是武汉大学计算机学院软件工程研究生课程"需求工程新技术"的期末结课大作业。KnowTrace 项目采用预训练模型微调的方法来进行需求代码追踪任务。

## 仓库结构

- `datasets/`: 存储数据集
  - 目前仅提供 iTrust 处理后的数据集
  - 其他相关数据集可从 [huiAlex/TAROT](https://github.com/huiAlex/TAROT) 下载
- `output/`: 存储微调后的预训练模型
- `sentence_transformers/`: 微调和评估所需的库
- `trace_eval.py`: 用于评估微调后的预训练模型在需求代码追踪任务中的表现
- `trace_pretrain.py`: 用于微调预训练模型

## 数据集说明

由于工作仍在进行中，为了保护研究成果，本仓库目前仅提供 iTrust 处理后的数据集。其他相关数据集可以从 [huiAlex/TAROT: Using Consensual Biterms from Text Structures of Requirements and Code to Improve IR-Based Traceability Recovery](https://github.com/huiAlex/TAROT) 下载。这也是我们所改进的 SOTA 方法的来源。

## 使用说明

1. 首先确保已安装所有必要的依赖，特别是 `sentence_transformers` 库。

2. 要微调预训练模型，运行：
   ```
   python trace_pretrain.py
   ```

3. 要评估微调后的模型在需求代码追踪任务中的表现，运行：
   ```
   python trace_eval.py
   ```

## 注意事项

- 在使用其他数据集时，请确保遵守相关的使用协议和版权规定。
- 微调过程可能需要较长时间，请耐心等待。
- 如有任何问题或建议，欢迎提出 issue 或 pull request。

## 致谢

感谢TAROT 项目的作者们提供的宝贵资源和研究基础。