---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- emotion-recognition
---


# 安装环境

```shell
pip install modelscope funasr>=1.0.0
```

# 用法

## 基于modelscope进行推理

```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.emotion_recognition,
    model="iic/emotion2vec_base", model_revision="v2.0.4")

rec_result = inference_pipeline('https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)
```


## 基于FunASR进行推理

```python
from funasr import AutoModel

model = AutoModel(model="iic/emotion2vec_base")

res = model(input='https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav', output_dir="./outputs")
print(res)
```
注：模型会自动下载

支持输入文件列表，wav.scp（kaldi风格）：
```cat wav.scp
wav_name1 wav_path1.wav
wav_name2 wav_path2.wav
...
```

输出为情感表征向量，保存在`output_dir`中，格式为numpy格式（可以用np.load()加载）

# 说明

本仓库为emotion2vec的modelscope版本，模型参数完全一致。

原始仓库地址: [https://github.com/ddlBoJack/emotion2vec](https://github.com/ddlBoJack/emotion2vec)

modelscope版本仓库：[https://github.com/alibaba-damo-academy/FunASR](https://github.com/alibaba-damo-academy/FunASR/tree/funasr1.0/examples/industrial_data_pretraining/emotion2vec)

# 相关论文以及引用信息
```BibTeX
@article{ma2023emotion2vec,
  title={emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation},
  author={Ma, Ziyang and Zheng, Zhisheng and Ye, Jiaxin and Li, Jinchao and Gao, Zhifu and Zhang, Shiliang and Chen, Xie},
  journal={arXiv preprint arXiv:2312.15185},
  year={2023}
}
```