from django.http import HttpRequest, JsonResponse
from hashlib import sha1

from api.wav_analyzer.analyze import fs_to_notes, plot_frequencies, wav_to_fs
from api.wav_analyzer.wav import remove_wav_headers


def analyze(request: HttpRequest):
    chunk = request.body
    remove_wav_headers(chunk)
    x, y = wav_to_fs(chunk)

    hash = sha1(chunk).hexdigest()[:8]
    plot_file = f'api\\static\\api\\plots\\{hash}.png'
    plot_frequencies(x, y, plot_file)

    return JsonResponse({
        'notes': fs_to_notes(x, y),
        'plot': plot_file[len("api\\"):]
    })