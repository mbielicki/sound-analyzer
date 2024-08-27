import * as React from "react";
import AudioRecorder from "./AudioRecorder";
import Console from "./Console";

export default function App() {
    return (
        <main className="w-screen h-screen bg-slate-800
        flex flex-col items-center gap-5 pt-4">
            <h1 className="font-bold text-3xl w-full 
            text-center text-white">Sound analyzer</h1>
            <AudioRecorder />
            <Console />
        </main>
    )
}