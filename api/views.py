from django.shortcuts import render
from django.http import HttpRequest, JsonResponse

from api.wav_analyzer.Analyzer import Analyzer
from api.wav_analyzer.utils import normalize_chunk

# Create your views here.

def analyze(request: HttpRequest):
    chunk = request.body
    normalize_chunk(chunk)
    analyzer = Analyzer(chunk)

    analyzer.plot()

    return JsonResponse({
        'notes': analyzer.extract_notes(),
        'plot': analyzer.plot_file
    })