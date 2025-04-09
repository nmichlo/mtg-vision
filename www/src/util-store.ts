import { atom } from 'nanostores';
import type {Detection, Device, Stats} from "./types";

export const $devices = atom<Device[]>([]);
export const $selectedDevice = atom<string|null>(null);
export const $isStreaming = atom<boolean>(false);
export const $detections = atom<Detection[]>([]);
export const $selectedId = atom<number|null>(null);
export const $status = atom<string>('Click "Start Streaming" to begin.');
export const $stats = atom<Stats>({messagesSent: 0, messagesReceived: 0});
