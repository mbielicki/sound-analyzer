import * as React from "react";

type key = {
    id: number,
    type: string
}

const createKeys = () => {
    const keys: key[] = Array()

    let i = 0
    const pushKeys = (colors: string[]) => {
        for (const color of colors) {
            console.log(color)
            keys.push({ id: i++, type: color })
        }
    }

    pushKeys(['white', 'black', 'white']) // A0 - B0

    for (let j = 0; j < 7 * 12; j++) {
        if ([1, 3, 6, 8, 10].includes(j % 12))
            pushKeys(['black'])
        else
            pushKeys(['white'])
    }

    return keys
}

const keys = createKeys()
console.log(keys)

const Piano = ({ pressedKeys = [] }) => {
    const getKeyClasses = (key: key) => {
        const baseClasses = "border border-black relative";
        const colorClasses = pressedKeys.includes(key.id) ? "bg-green-500" :
            (key.type === 'white' ? "bg-white" : "bg-black");
        const sizeClasses = key.type === 'white' ? "w-5 h-16" : "w-2 h-10 -mx-1 z-10";

        return `${baseClasses} ${colorClasses} ${sizeClasses}`;
    };

    return (
        <div className="flex justify-center p-5 overflow-x-auto">
            {keys.map((key) => (
                <div
                    key={key.id}
                    className={getKeyClasses(key)}
                />
            ))}
        </div>
    );
};


export default Piano;