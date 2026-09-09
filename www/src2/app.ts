/**
 * Application Entry Point
 *
 * Main entry point for the MTG Vision web application. This file contains:
 * - The main() function that initializes the application
 * - Creation of the VideoContainer component
 * - Loading of the card identification index
 *
 * This is the top-level module that brings together all components
 * and services to create the functioning application.
 */

import { VideoContainer } from "./component-video";
import { loadIndex } from "./util-index";

/**
 * Initialize the Magic Card Vision application
 */
export function main() {
  // Create and mount the main video container component
  if (!customElements.get("video-container")) {
    customElements.define("video-container", VideoContainer);
  }
  document.querySelector("video-container")?.remove();
  const videoContainer = document.createElement("video-container");
  document.body.appendChild(videoContainer);

  // Load the card database index
  loadIndex()
    .then((index) => {
      console.log("Card index loaded successfully");
    })
    .catch((error) => {
      console.error("Error loading card index:", error);
    });
}
