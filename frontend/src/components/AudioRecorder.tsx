import * as React from "react";
import MicIcon from '@mui/icons-material/Mic';
import { useEffect, useState } from "react";
import { MicController, ProcessAudio } from "../utils/MicController";

const processAudio: ProcessAudio = (data) => {
    // console.log(`Audio data length: ${data.size}`)
}
const micController = new MicController();

export default function AudioRecorder() {
    const [micIsOn, setMicIsOn] = useState(false)

    useEffect(() => {
        micController.setup(processAudio)
            .then(() => setMicIsOn(true))
            .catch(console.error)
    }, [])

    const onClickMicIcon: React.MouseEventHandler<SVGSVGElement>
        = e => {
            if (micIsOn) {
                setMicIsOn(false);
                micController.stop()
            } else {
                setMicIsOn(true);
                micController.start()
            }
        }

    return (
        <MicIcon fontSize="large" onClick={onClickMicIcon}
            className={
                (micIsOn ? "text-red-700" : "text-white")
                + " hover:opacity-75 active:text-slate-900"
            }
        />
    )

}