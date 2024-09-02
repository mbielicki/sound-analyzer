from django.http import HttpRequest, JsonResponse
from api.wav_analyzer.analyze import extract_frequencies, plot_frequencies, extract_notes, remove_wav_headers
from hashlib import sha1


def analyze(request: HttpRequest):
    chunk = request.body
    remove_wav_headers(chunk)
    x, y = extract_frequencies(chunk)

    hash = sha1(chunk).hexdigest()[:8]
    plot_file = f'api\\static\\api\\plots\\{hash}.png'
    plot_frequencies(x, y, plot_file)

    return JsonResponse({
        'notes': extract_notes(x, y),
        'plot': plot_file[len("api\\"):]
    })