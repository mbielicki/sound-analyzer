import * as React from "react";
import AudioRecorder from "./AudioRecorder";
import Plot from "./Plot";
import { useState } from "react"

export default function App() {
    const [plt, setPlt] = useState("")

    return (
        <main className="min-h-screen
        flex flex-col items-center gap-5 py-4">
            <h1 className="font-bold text-3xl text-white">Sound analyzer</h1>
            <AudioRecorder onAnalyzed={(res) => setPlt(res.data.plot)} />
            <Plot plotFile={plt} />
        </main>
    )
}