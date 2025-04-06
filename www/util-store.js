import { atom } from 'https://esm.run/nanostores';

/**
 * @typedef {Object} Device
 * @property {string} deviceId - Unique identifier for the device
 * @property {string} label - Human-readable name of the device
 * @property {string} kind - Type of device (e.g., 'videoinput')
 */

/**
 * @typedef {Object} Match
 * @property {string} name - Name of the matched card
 * @property {string} [set_name] - Optional set name of the card
 * @property {string} [set_code] - Optional set code of the card
 * @property {string} [img_uri] - Optional URI of the card’s image
 */

/**
 * @typedef {Object} Detection
 * @property {number} id - Unique identifier for the detection
 * @property {string} color - Color associated with the detection
 * @property {number[][]} points - Array of [x, y] coordinates for the detection polygon
 * @property {string} img - Base64-encoded image of the detected area
 * @property {Match[]} matches - Array of potential matches for the detection
 */

/**
 * @template T
 * @typedef {Object} Atom
 * @property {function(): T} get - Get the current value of the atom
 * @property {function(T): void} set - Set a new value for the atom
 * @property {function(function(T): void): function(): void} subscribe - Subscribe to changes, returns an unsubscribe function
 * @property {function(function(T): void): function(): void} listen - Listen to changes, returns an unsubscribe function
 */

/** @type {Atom<Device[]>} */
export const $devices = atom([]);
/** @type {Atom<string|null>} */
export const $selectedDevice = atom(null);
/** @type {Atom<boolean>} */
export const $isStreaming = atom(false);
/** @type {Atom<Detection[]>} */
export const $detections = atom([]);
/** @type {Atom<number|null>} */
export const $selectedId = atom(null);
/** @type {Atom<string>} */
export const $status = atom('Click "Start Streaming" to begin.');

/**
 * @typedef {Object} Stats
 * @property {number} messagesSent
 * @property {string} messagesReceived
 */

/** @type {Atom<Stats>} */
export const $stats = atom({messagesSent: 0, messagesReceived: 0});
