# Minstrel

Minstrel 是一个多智能体系统，用于生成基于LangGPT格式的结构化提示。该项目旨在通过多个智能代理协作生成高质量的LangGPT提示，以提高生成文本的准确性和多样性。

## News
- Minstrel正在参与书生大模型训练营第4期
  ![Minstrel正在参加书生大模型训练营第4期](https://github.com/user-attachments/assets/5a32bd82-e2fd-4bdb-81ea-d5ff9fb648dd)

## 功能特性

- 多代理协作生成LangGPT提示
- 高效的提示生成算法
- 易于扩展和定制

## 安装

请按照以下步骤安装和运行该项目：

1. 克隆项目仓库：
    ```bash
    git clone https://github.com/sci-m-wang/LangGPT-generator.git
    cd LangGPT-generator
    ```

2. 创建并激活虚拟环境（可选但推荐）：
    ```bash
    conda create -n langgpt python=3.10 -y
    conda activate langgpt
    ```

3. 安装依赖项：
    ```bash
    pip install openai==1.37.1
    pip install streamlit==1.37.0
    ```

## 使用方法

以下是一个简单的使用示例：

1. 运行主脚本以生成LangGPT提示：
    ```bash
    python -m streamlit run app.py
    ```

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建一个新的分支 (`git checkout -b feature-branch`)
3. 提交您的更改 (`git commit -am 'Add new feature'`)
4. 推送到分支 (`git push origin feature-branch`)
5. 创建一个新的 Pull Request

## 引用
如果您在研究中使用了本项目，请引用以下论文：
```
@misc{wang2024langgptrethinkingstructuredreusable,
      title={LangGPT: Rethinking Structured Reusable Prompt Design Framework for LLMs from the Programming Language}, 
      author={Ming Wang and Yuanzhong Liu and Xiaoyu Liang and Songlian Li and Yijie Huang and Xiaoming Zhang and Sijia Shen and Chaofeng Guan and Daling Wang and Shi Feng and Huaiwen Zhang and Yifei Zhang and Minghui Zheng and Chi Zhang},
      year={2024},
      eprint={2402.16929},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2402.16929}, 
}

@misc{wang2024minstrelstructuralpromptgeneration,
      title={Minstrel: Structural Prompt Generation with Multi-Agents Coordination for Non-AI Experts}, 
      author={Ming Wang and Yuanzhong Liu and Xiaoyu Liang and Yijie Huang and Daling Wang and Xiaocui Yang and Sijia Shen and Shi Feng and Xiaoming Zhang and Chaofeng Guan and Yifei Zhang},
      year={2024},
      eprint={2409.13449},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.13449}, 
}
```

## 联系方式

如果您有任何问题或建议，请通过以下方式联系我们：

- 电子邮件: sci.m.wang@gmail.com
- GitHub: [sci-m-wang](https://github.com/sci-m-wang)

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=sci-m-wang/Minstrel&type=Date)

感谢您使用 LangGPT-generator！
