export const baseUrl = "http://127.0.0.1:8000";

export const mediaRecorderOptions = {
    mimeType: "audio/webm;codecs=opus",
    audioBitsPerSecond: 128_000,
    audioBitrateMode: "constant",
};

export const mediaRecorderTimeslice = 1000;
export const newAudioFileName = "audio-analyzed.webm";
