from mser.predict import MSERPredictor

def analyze_audio(path):
    # 获取识别器
    predictor = MSERPredictor(configs='E:/study/SpeechEmotionPlatform/EmotionRecognitionPlatform/audio/recognition/configs/bi_lstm.yml', use_ms_model=None,
                              model_path='E:/study/SpeechEmotionPlatform/EmotionRecognitionPlatform/audio/recognition/models/BiLSTM_Emotion2Vec/best_model/', use_gpu=True)

    # 对指定音频进行预测
    label, score = predictor.predict(audio_data=path)

    # 打印预测结果
    print(f'预测结果标签为：{label}，得分：{score}')

    result = f'预测结果标签为：{label}，得分：{score}'

    return result