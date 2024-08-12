import * as React from "react";
import MicIcon from '@mui/icons-material/Mic';
import DownloadIcon from '@mui/icons-material/Download';
import { useEffect, useState } from "react";
import { MicController, ProcessAudio } from "../utils/MicController";
import axios from 'axios';
import { baseUrl } from '../config'
const Cookies = require('js-cookie')

const processAudio: ProcessAudio = (data) => {
    data.arrayBuffer().then(buffer => {
        console.log(buffer)
        axios.post(baseUrl + '/api/analyze', {
            data: JSON.stringify(Array.from(new Uint8Array(buffer))),

        }, {
            headers: {
                'X-CSRFToken': Cookies.get('csrftoken')
            }
        })
            .then(res => {
                console.log(res)
            })
    })
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

    return (<>
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
    </>
    )

}