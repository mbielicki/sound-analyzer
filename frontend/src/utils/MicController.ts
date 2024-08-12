import { mediaRecorderOptions, mediaRecorderTimeslice, newAudioFileName } from "../config"

class AudioRecorderError extends Error {
    constructor(message: string) {
        super(message)
        this.name = "AudioRecorderError"
    }
}

export type ProcessAudio = (e: Blob) => void
 
export class MicController {
    audioBlobs: Blob[]
    recorder: MediaRecorder
    start: () => void
    stop: () => void

    constructor() {
        this.audioBlobs = []
        this.recorder = null
    }

    async setup(processAudio: ProcessAudio) {
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia))
            return Promise.reject(new AudioRecorderError('mediaDevices API or getUserMedia method is not supported in this browser.'))

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        this.recorder = new MediaRecorder(stream, mediaRecorderOptions)
        this.recorder.ondataavailable = e => {
            processAudio(e.data)
            this.audioBlobs.push(e.data)
        }

        this.recorder.onstop = e => {
            const fullAudio = new Blob(this.audioBlobs)
            this.audioBlobs = [];
            const audioURL = window.URL.createObjectURL(fullAudio)
            const a = document.querySelector('a#newAudioFileDownload')
            if (a instanceof HTMLAnchorElement) {
                a.classList.remove('hidden')
                a.classList.add('inline')
                a.href = audioURL
                a.download = newAudioFileName
                a.onclick = e => {
                    a.classList.remove('inline')
                    a.classList.add('hidden')
                }
            } else
                console.error('Cannot find anchor element to download the recorded audio.')
        }

        this.start = () => this.recorder.start(mediaRecorderTimeslice)
        this.stop = this.recorder.stop.bind(this.recorder)

        this.start()
    }
    
}