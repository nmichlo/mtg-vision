

export interface Device {
    deviceId: string;
    label: string;
    kind: string;
}

export interface Match {
    id: string;
    name: string;
    score: number

    set_name?: string;
    set_code?: string;
    img_uri?: string;

    all_data?: object
}

export interface Detection {
    id: number;
    color: string;
    points: number[][];
    img: string;
    matches: Match[];
}

export interface Stats {
    messagesSent: number;
    messagesReceived: number;
}


export type SvgInHtml = HTMLElement & SVGElement;
