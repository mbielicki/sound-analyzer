import * as React from "react";
import MicIcon from '@mui/icons-material/Mic';
import DownloadIcon from '@mui/icons-material/Download';
import { useEffect, useState } from "react";
import { MicController, ProcessAudio } from "../utils/MicController";
import axios from 'axios';
import { baseUrl } from '../config'
import Piano from "./Piano";
const Cookies = require('js-cookie')


const micController = new MicController();

export default function AudioRecorder() {
    const [micIsOn, setMicIsOn] = useState(false)
    const [pressedKeys, setPressedKeys] = useState([])

    const postAudio: ProcessAudio = (data) => {
        data.arrayBuffer().then(arrBuffer => {

            const audioData = arrBuffer

            axios.post(baseUrl + '/api/analyze', audioData, {
                headers: {
                    'X-CSRFToken': Cookies.get('csrftoken')
                }
            })
                .then(res => {
                    setPressedKeys(res.data.notes)
                })
        })
    }

    useEffect(() => {
        micController.setup(postAudio)
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
        <div className="flex flex-col items-center justify-center">
            <MicIcon fontSize="large" onClick={onClickMicIcon}
                className={
                    (micIsOn ? "text-red-700" : "text-white")
                    + " hover:opacity-75 active:text-slate-900"
                }
            />
            <a id="newAudioFileDownload" className="hidden" >
                <DownloadIcon fontSize="large"
                    className="text-white hover:opacity-75 active:text-slate-900" />
            </a>
            <Piano pressedKeys={pressedKeys} />
        </div>
    )

}