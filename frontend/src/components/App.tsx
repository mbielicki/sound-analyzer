import * as React from "react";
import MicIcon from '@mui/icons-material/Mic';


export default function App() {
    return (
        <main className="w-screen h-screen bg-slate-800
        flex flex-col items-center gap-5 pt-4">
            <h1 className="font-bold text-3xl w-full 
            text-center text-white">Sound analyzer</h1>
            <MicIcon fontSize="large" className="text-white" />
            <div className="w-3/4 h-40 bg-slate-400"></div>
            <div className="w-3/4 h-40 bg-slate-400"></div>
            <div className="w-3/4 h-40 bg-slate-400"></div>
        </main>
    )
}