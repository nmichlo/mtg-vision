

export interface Device {
    deviceId: string;
    label: string;
    kind: string;
}

export interface Match {
    name: string;
    set_name?: string;
    set_code?: string;
    img_uri?: string;
    type_line?: string;
    price?: string;
    oracle_text?: string;
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
