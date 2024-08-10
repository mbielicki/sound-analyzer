import * as React from "react";
import MicIcon from '@mui/icons-material/Mic';
import { useEffect } from "react";
import { MicController, ProcessAudio } from "../utils/MicController";

const processAudio: ProcessAudio = (data) => {
    console.log(`Audio data length: ${data.size}`)
}

export default function AudioRecorder() {
    useEffect(() => {
        const micController = new MicController();
        micController.start(processAudio).catch(console.error);
    });
    return (
        <MicIcon fontSize="large" className="text-white" />
    )

}