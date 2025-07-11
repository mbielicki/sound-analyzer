export const baseUrl = "http://192.168.0.107:8000";
// export const baseUrl = "http://localhost:8000";

export const mediaRecorderOptions = {
    mimeType: "audio/wav",
    audioBitsPerSecond: 48_000 * 16,
    audioBitrateMode: "constant",
};

export const mediaRecorderTimeslice = 300;
export const newAudioFileName = "audio-analyzed.wav";
