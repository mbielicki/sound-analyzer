const TIMESLICE = 1000

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

        this.recorder = new MediaRecorder(stream)
        this.recorder.ondataavailable = e => {
            processAudio(e.data)
            this.audioBlobs.push(e.data)
        }

        this.recorder.onstop = e => {
            // const fullAudio = new Blob(this.audioBlobs)
            // this.audioBlobs = [];
            // const audioURL = window.URL.createObjectURL(fullAudio)
            // const a = document.createElement('a')
            // a.style.display = 'none'
            // a.href = audioURL
            // a.download = 'audio-analyzed.wav'
            // document.body.appendChild(a)
            // a.click()
            // setTimeout(() => {
            //     document.body.removeChild(a)
            //     window.URL.revokeObjectURL(audioURL)
            // }, 1000)
        }

        this.start = () => this.recorder.start(TIMESLICE)
        this.stop = this.recorder.stop.bind(this.recorder)

        this.recorder.start(TIMESLICE)
    }
    
}