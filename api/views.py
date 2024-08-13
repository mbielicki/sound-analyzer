from django.shortcuts import render
from django.http import HttpRequest, JsonResponse

from api.wav_analyzer.Analyzer import Analyzer
from api.wav_analyzer.utils import save_wav

# Create your views here.

def analyze(request: HttpRequest):
    chunk = request.body
    analyzer = Analyzer()

    return JsonResponse({
        'notes': analyzer.extract_notes(chunk)
    })