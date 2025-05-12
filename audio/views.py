import json

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

from .recognition.analysis import analyze_audio

# Create your views here.
def upload_file(request):
    analysis_results_display = None
    uploaded_filename = None
    error_msg = None

    if request.method == 'POST':
        uploaded_file = request.FILES.get('audio_file')

        if uploaded_file:
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)

            try:
                #保存文件
                filename = fs.save(uploaded_file.name, uploaded_file)
                file_path = os.path.join(settings.MEDIA_ROOT, filename)
                uploaded_filename = uploaded_file.name

                print(f'准备分析文件{file_path}')
                analysis_results = analyze_audio(file_path)
                print(f'分析结果{analysis_results}')

                try:
                    analysis_results_display = json.dumps(analysis_results, indent=2, ensure_ascii=False)
                except(TypeError, OverflowError):
                    analysis_results_display = str(analysis_results)

            except Exception as e:
                print(e)
                error_msg = e

        else:
            error_msg = '请选择一个音频上传'

        context = {
            'results': analysis_results_display,
            'filename': uploaded_filename,
            'error': error_msg,
        }

        return render(request, 'index.html', context)

    else:
        return render(request, 'index.html', context={})