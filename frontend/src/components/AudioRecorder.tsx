import * as React from "react";
import MicIcon from '@mui/icons-material/Mic';
import { useEffect } from "react";

class AudioRecorderError extends Error {
    constructor(message: string) {
        super(message);
        this.name = "AudioRecorderError";
    }
}

const TIMESLICE = 1000;

class MicController {
    audioBlobs: Blob[]
    recorder: MediaRecorder

    constructor() {
        this.audioBlobs = [];
        this.recorder = null;
    }

    async start() {
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia))
            return Promise.reject(new AudioRecorderError('mediaDevices API or getUserMedia method is not supported in this browser.'));

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        this.recorder = new MediaRecorder(stream);
        this.recorder.ondataavailable = e => {
            this.audioBlobs.push(e.data);


        };

        this.recorder.onstop = e => {
            const fullAudio = new Blob(this.audioBlobs);
            this.audioBlobs = [];
            const audioURL = window.URL.createObjectURL(fullAudio);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = audioURL;
            a.download = 'audio-analyzed.wav';
            document.body.appendChild(a);
            a.click();
            setTimeout(() => {
                document.body.removeChild(a);
                window.URL.revokeObjectURL(audioURL);
            }, 1000);
        };

        this.recorder.start(TIMESLICE)
    }


}



export default function AudioRecorder() {
    useEffect(() => {
        const micController = new MicController();
        micController.start().catch(console.error);
    });
    return (
        <MicIcon fontSize="large" className="text-white" />
    )

}