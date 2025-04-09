

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

    all_data?: any
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
    processTime: number | null;
}

export interface Payload {
    detections: Detection[];
    process_time: number; // seconds
    send_time: number; // seconds
}


export type SvgInHtml = HTMLElement & SVGElement;
