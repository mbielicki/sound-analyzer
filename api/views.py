from django.http import HttpRequest, JsonResponse

from api.wav_analyzer.analyze import extract_frequencies, plot_frequencies, extract_notes
from api.wav_analyzer.utils import normalize_chunk
from hashlib import sha1


def analyze(request: HttpRequest):
    chunk = request.body
    normalize_chunk(chunk)
    yf = extract_frequencies(chunk)

    hash = sha1(chunk).hexdigest()[:8]
    plot_file = f'api\\static\\api\\plots\\{hash}.png'
    plot_frequencies(yf, plot_file)

    return JsonResponse({
        'notes': extract_notes(yf),
        'plot': plot_file[len("api\\"):]
    })