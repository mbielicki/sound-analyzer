import * as React from "react"
import { baseUrl } from '../config'

type Props = {
    plotFile: string;
}
export default function Plot({ plotFile }: Props) {
    const url = baseUrl + '/' + plotFile

    return (
        <img src={url} alt="plot" />
    )
}
