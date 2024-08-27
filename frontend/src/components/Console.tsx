import * as React from "react";
import { useEffect, useState, useRef } from "react";

const Console = () => {
    const [logs, setLogs] = useState([]);

    useEffect(() => {
        const oldLog = console.log;
        var nextId = 0

        console.log = function (...items) {

            // Call native method first
            oldLog.apply(this, items);

            // Use JSON to transform objects, all others display normally
            items.forEach((item, i) => {
                items[i] = (typeof item === 'object' ? JSON.stringify(item, null, 4) : item);
            });
            const message = items.join(' ')
            setLogs(prev => [...prev, { id: nextId++, message: message }])

        };

    }, [])

    const divRef = useRef(null)
    useEffect(() => {
        divRef.current.scrollTop = divRef.current.scrollHeight;
    })

    return (
        <div className="w-3/4 h-80 bg-slate-900 text-white p-1 overflow-auto" ref={divRef}>

            <ul>
                {logs.map(log => (
                    <li className="p-1" key={log.id}>{log.message}</li>
                ))}
            </ul>
        </div>
    )
}

export default Console