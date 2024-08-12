from django.shortcuts import render
from django.http import HttpRequest, JsonResponse

# Create your views here.

def analyze(request: HttpRequest):
    return JsonResponse({
        'notes': ['c4', 'e4', 'g4']
    })